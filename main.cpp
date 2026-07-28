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
void LocatePeriodicPos(float* wav, std::vector<double>& periodPos, int blockSize, int numSamples)
{
	std::vector<float> buf;
	std::vector<float> corr;
	buf.resize(blockSize, 0);
	corr.resize(blockSize, 0);
	for (int i = 0; i < blockSize; ++i) buf[i] = wav[i];

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

	int len2t = sampleRate / basefreq * 2;//2个周期长度
	std::vector<double> periodPosl{ 0 };
	std::vector<double> periodPosr{ 0 };
	LocatePeriodicPos(wavl, periodPosl, len2t, numSamples);
	LocatePeriodicPos(wavr, periodPosr, len2t, numSamples);
	
}