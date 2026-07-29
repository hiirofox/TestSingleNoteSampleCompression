#include <string>
#include <vector>
#include "wavfile.h"

float sampleRate = 44100.0;
float basefreq = 130.81;//Hz

//三点插值提取峰
double ParabolicPeak(double ym, double y0, double yp)
{
	double d = ym - 2.0 * y0 + yp;

	if (fabs(d) < 1e-12)
		return 0.0;

	double delta = 0.5 * (ym - yp) / d;

	if (delta < -0.5) delta = -0.5;
	if (delta > 0.5) delta = 0.5;

	return delta;
}

//查找周期坐标
void LocatePeriodicPos(std::vector<float>& buf, int isPinSearchBuf,
	float* wav, std::vector<double>& periodPos, int blockSize, int numSamples)
{
	std::vector<float> corr;
	corr.resize(blockSize, 0);
	if (!isPinSearchBuf) buf.resize(blockSize, 0);
	if (!isPinSearchBuf) for (int i = 0; i < blockSize; ++i) buf[i] = wav[i];

	int step = blockSize;//步进大小，补偿因音高偏移而漏掉或多处理的块
	for (int i = 0; i < numSamples; i += step)
	{
		int searchStart = i + blockSize / 2;
		if (searchStart + 2 * blockSize > numSamples) break;
		for (int j = 0; j < blockSize; ++j)//计算非标准的互相关
		{
			float cv = 0;
			for (int k = 0; k < blockSize; ++k)
			{
				int idx = searchStart + j + k;
				float x1 = buf[k];
				float x2 = wav[idx];
				float y = x1 - x2;
				cv += y * y;//(x1-x2)^2
			}
			corr[j] = cv;
		}
		float maxcorr = std::numeric_limits<float>::max();
		int maxcorrPos = blockSize / 2;
		for (int j = 0; j < blockSize; ++j)//寻找最大相关位置
		{
			if (maxcorr > corr[j])//对于(x1-x2)^2，应该找谷，就是最小值处是最大“相关”
			{
				maxcorr = corr[j];
				maxcorrPos = j;
			}
		}
		//printf("%d\n", maxcorrPos);

		//根据最大相关相邻相关值重新分配最大相关小数位置
		double mcposf = maxcorrPos;
		if (maxcorrPos >= 1 && maxcorrPos < blockSize - 1)
		{
			mcposf += ParabolicPeak(corr[maxcorrPos - 1], corr[maxcorrPos], corr[maxcorrPos + 1]);
		}
		//写入数组
		double globalPos = i + blockSize / 2 + mcposf;
		periodPos.push_back(globalPos);

		step = blockSize / 2 + mcposf;//更新下次搜索的步进大小
		if (!isPinSearchBuf)
		{
			int nextBufAt = (float)i + blockSize / 2 + mcposf;
			float frac = mcposf - (int)mcposf;
			for (int j = 0; j < blockSize; ++j)//更新下一次查找用的块
			{
				float a1 = wav[nextBufAt + j];
				float a2 = wav[nextBufAt + j + 1];
				buf[j] = a1 + (a2 - a1) * frac;
			}
		}
	}
}


struct CompressionSamples
{
	int numPeriods = 0;
	int blockSize = 0;
	std::vector<double> periodPos;
	std::vector<float> aVal;
	std::vector<float> wavetable;
	int GetNumSamples() { return periodPos[periodPos.size() - 1] + blockSize; }
	float GetSample(float posf)
	{
		int pos1 = ((int)posf) % blockSize;
		int pos2 = (pos1 + 1) % blockSize;
		float frac = posf - pos1;
		float a1 = wavetable[pos1];
		float a2 = wavetable[pos1];
		return a1 + (a2 - a1) * frac;
	}
	void Resynth(std::vector<float>& out)
	{
		int numSamples = GetNumSamples();
		out.resize(numSamples, 0);
		for (int i = 0; i < numPeriods; ++i)
		{
			int startPos = periodPos[i];
			for (int j = 0; j < blockSize; ++j)
			{
				out[startPos + j] += wavetable[j] * aVal[i];
			}
		}
	}
};

CompressionSamples TestCompression(float* wav, int blockSize, int numSamples)
{
	std::vector<float> searchBuf;
	std::vector<double> periodPos{ 0 };
	LocatePeriodicPos(searchBuf, false, wav, periodPos, blockSize, numSamples);//先解一遍周期标记位置

	std::vector<float> wavetable;
	wavetable.resize(blockSize, 0);//平均化采样
	for (double pos : periodPos)
	{
		int start = pos;
		float frac = pos - (int)pos;
		for (int i = 0; i < blockSize; ++i)
		{
			float a1 = wav[start + i + 0];
			float a2 = wav[start + i + 1];
			wavetable[i] += a1 + (a2 - a1) * frac;
		}
	}
	for (float& v : wavetable) v /= periodPos.size();
	//把波表改成首尾能够相连的形式

	auto tmpwt = wavetable;
	auto window = [&](float x) {return 0.5 - 0.5 * cosf(x * 2.0 * 3.1415926535897932384626 / blockSize); };
	for (int i = 0; i < blockSize; ++i)
	{
		float v = tmpwt[i] * window(i);
		wavetable[i] = v;
	}
	tmpwt = wavetable;

	//再用波表搜一遍周期坐标
	periodPos.clear();
	LocatePeriodicPos(tmpwt, false, wav, periodPos, blockSize, numSamples);

	int numPeriods = periodPos.size();
	std::vector<float> va;//波表应用到目标位置的幅值
	va.resize(numPeriods, 0);
	float avgWT = 0;//波表的平均值
	for (float v : wavetable)avgWT += v;
	avgWT /= blockSize;
	for (int i = 0; i < numPeriods; ++i)
	{
		int pos = std::round(periodPos[i]);
		float avgWav = 0;//目标周期的平均值
		for (int j = 0; j < blockSize; ++j)avgWav += wav[pos + j];
		avgWav /= blockSize;
		float numSum = 0;
		float denSum = 0;
		for (int j = 0; j < blockSize; ++j)
		{
			float vwt = wavetable[j];
			float vwav = wav[pos + j];
			numSum += (vwt - avgWT) * (vwav - avgWav);
			denSum += (vwt - avgWT) * (vwt - avgWT);
		}
		va[i] = numSum / denSum;//波表在目标周期处的增益
	}

	return { numPeriods,blockSize,periodPos,va,wavetable };
}

int main()
{
	WavReader wr;
	wr.OpenWAV("D:/Projects/c++/TestAudioCompression/C3.wav");
	int numSamples = wr.GetNumSamples();
	printf("wav numSamples:%d\n", numSamples);
	float* wavl = new float[numSamples + 1000];
	float* wavr = new float[numSamples + 1000];
	memset(wavl, 0, sizeof(float) * numSamples);
	memset(wavr, 0, sizeof(float) * numSamples);
	wr.ReadBlock(wavl, wavr, numSamples);//读取wav

	int blockSize = sampleRate / basefreq * 2;//周期长度
	//int blockSize = 1024;
	auto cmpl = TestCompression(wavl, blockSize, numSamples);
	auto cmpr = TestCompression(wavl, blockSize, numSamples);

	int bufSamples = (std::min)(cmpl.GetNumSamples(), cmpr.GetNumSamples());
	printf("bufSamples:%d\n", bufSamples);
	std::vector<float> scl, scr;
	cmpl.Resynth(scl);
	cmpr.Resynth(scr);

	WavWriter wo;
	wo.CreateWAV("D:/Projects/c++/TestAudioCompression/Resynth.wav", sampleRate);
	wo.WriteBlock(scl.data(), scr.data(), bufSamples);
}