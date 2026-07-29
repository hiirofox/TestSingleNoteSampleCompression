#pragma once

#include <fstream>
#include <string>
#include <cstdint>
#include <algorithm>

class WavWriter
{
public:
	void CreateWAV(std::string path, float sampleRate = 48000.0f)
	{
		Close();

		m_sampleRate = static_cast<uint32_t>(sampleRate);
		m_dataBytes = 0;

		m_file.open(path, std::ios::binary);
		if (!m_file.is_open())
			return;

		WriteHeader();
	}

	void WriteBlock(const float* bufl, const float* bufr, int numSamples)
	{
		if (!m_file.is_open() || bufl == nullptr || bufr == nullptr || numSamples <= 0)
			return;

		for (int i = 0; i < numSamples; ++i)
		{
			int16_t l = FloatToInt16(bufl[i]);
			int16_t r = FloatToInt16(bufr[i]);

			WriteValue(l);
			WriteValue(r);

			m_dataBytes += sizeof(int16_t) * 2;
		}
	}

	void Close()
	{
		if (!m_file.is_open())
			return;

		// 回填 RIFF chunk size
		m_file.seekp(4, std::ios::beg);
		uint32_t riffSize = 36 + m_dataBytes;
		WriteValue(riffSize);

		// 回填 data chunk size
		m_file.seekp(40, std::ios::beg);
		WriteValue(m_dataBytes);

		m_file.close();
	}

	~WavWriter()
	{
		Close();
	}

private:
	std::ofstream m_file;
	uint32_t m_sampleRate = 48000;
	uint32_t m_dataBytes = 0;

	static int16_t FloatToInt16(float v)
	{
		v = std::clamp(v, -1.0f, 1.0f);
		return static_cast<int16_t>(v * 32767.0f);
	}

	template <typename T>
	void WriteValue(T value)
	{
		m_file.write(reinterpret_cast<const char*>(&value), sizeof(T));
	}

	void WriteText(const char* text, int size)
	{
		m_file.write(text, size);
	}

	void WriteHeader()
	{
		const uint16_t numChannels = 2;
		const uint16_t bitsPerSample = 16;
		const uint16_t audioFormat = 1; // PCM
		const uint32_t byteRate = m_sampleRate * numChannels * bitsPerSample / 8;
		const uint16_t blockAlign = numChannels * bitsPerSample / 8;

		WriteText("RIFF", 4);
		WriteValue<uint32_t>(0); // 稍后回填
		WriteText("WAVE", 4);

		WriteText("fmt ", 4);
		WriteValue<uint32_t>(16); // fmt chunk size
		WriteValue<uint16_t>(audioFormat);
		WriteValue<uint16_t>(numChannels);
		WriteValue<uint32_t>(m_sampleRate);
		WriteValue<uint32_t>(byteRate);
		WriteValue<uint16_t>(blockAlign);
		WriteValue<uint16_t>(bitsPerSample);

		WriteText("data", 4);
		WriteValue<uint32_t>(0); // 稍后回填
	}
};

class WavReader
{
public:
	bool OpenWAV(std::string path)
	{
		Close();

		m_file.open(path, std::ios::binary);
		if (!m_file.is_open())
			return false;

		return ReadHeader();
	}

	int ReadBlock(float* bufl, float* bufr, int numSamples)
	{
		if (!m_file.is_open() || bufl == nullptr || bufr == nullptr || numSamples <= 0)
			return 0;

		int samplesRead = 0;

		for (int i = 0; i < numSamples; ++i)
		{
			if (m_dataBytesRemaining < sizeof(int16_t) * 2)
				break;

			int16_t l = 0;
			int16_t r = 0;

			ReadValue(l);
			ReadValue(r);

			if (!m_file)
				break;

			bufl[i] = Int16ToFloat(l);
			bufr[i] = Int16ToFloat(r);

			m_dataBytesRemaining -= sizeof(int16_t) * 2;
			++samplesRead;
		}

		// 文件末尾不足一整块时，剩余部分补 0
		for (int i = samplesRead; i < numSamples; ++i)
		{
			bufl[i] = 0.0f;
			bufr[i] = 0.0f;
		}

		return samplesRead;
	}

	void Close()
	{
		if (m_file.is_open())
			m_file.close();

		m_sampleRate = 0;
		m_numSamples = 0;
		m_dataBytesRemaining = 0;
		m_dataStartPos = 0;
	}

	bool IsOpen() const
	{
		return m_file.is_open();
	}

	uint32_t GetSampleRate() const
	{
		return m_sampleRate;
	}

	uint64_t GetNumSamples() const
	{
		return m_numSamples;
	}

	uint64_t GetSamplesRemaining() const
	{
		return m_dataBytesRemaining / (sizeof(int16_t) * 2);
	}

	~WavReader()
	{
		Close();
	}

private:
	std::ifstream m_file;

	uint32_t m_sampleRate = 0;
	uint64_t m_numSamples = 0;
	uint64_t m_dataBytesRemaining = 0;
	std::streampos m_dataStartPos = 0;

private:
	static float Int16ToFloat(int16_t v)
	{
		return static_cast<float>(v) / 32768.0f;
	}

	template <typename T>
	void ReadValue(T& value)
	{
		m_file.read(reinterpret_cast<char*>(&value), sizeof(T));
	}

	void ReadText(char* text, int size)
	{
		m_file.read(text, size);
	}

	static bool MatchText(const char* a, const char* b)
	{
		return a[0] == b[0] &&
			a[1] == b[1] &&
			a[2] == b[2] &&
			a[3] == b[3];
	}

	bool ReadHeader()
	{
		char riff[4];
		char wave[4];

		ReadText(riff, 4);

		uint32_t riffSize = 0;
		ReadValue(riffSize);

		ReadText(wave, 4);

		if (!m_file || !MatchText(riff, "RIFF") || !MatchText(wave, "WAVE"))
		{
			Close();
			return false;
		}

		bool foundFmt = false;
		bool foundData = false;

		uint16_t audioFormat = 0;
		uint16_t numChannels = 0;
		uint16_t bitsPerSample = 0;
		uint16_t blockAlign = 0;

		while (m_file && (!foundFmt || !foundData))
		{
			char chunkId[4];
			uint32_t chunkSize = 0;

			ReadText(chunkId, 4);
			ReadValue(chunkSize);

			if (!m_file)
				break;

			std::streampos chunkDataStart = m_file.tellg();

			if (MatchText(chunkId, "fmt "))
			{
				uint32_t byteRate = 0;

				ReadValue(audioFormat);
				ReadValue(numChannels);
				ReadValue(m_sampleRate);
				ReadValue(byteRate);
				ReadValue(blockAlign);
				ReadValue(bitsPerSample);

				foundFmt = true;
			}
			else if (MatchText(chunkId, "data"))
			{
				m_dataStartPos = chunkDataStart;
				m_dataBytesRemaining = chunkSize;

				foundData = true;
			}

			// 跳到下一个 chunk
			// 注意：即使找到了 data，也先记录 data 起点，最后再 seek 回来
			std::streamoff skipSize = static_cast<std::streamoff>(chunkSize);

			// WAV chunk 是 word-aligned，奇数字节后面有 1 byte padding
			if (skipSize % 2 == 1)
				++skipSize;

			m_file.seekg(chunkDataStart + skipSize, std::ios::beg);
		}

		if (!foundFmt || !foundData)
		{
			Close();
			return false;
		}

		// 这里只支持 16-bit PCM stereo
		if (audioFormat != 1 || numChannels != 2 || bitsPerSample != 16)
		{
			Close();
			return false;
		}

		if (blockAlign != sizeof(int16_t) * 2)
		{
			Close();
			return false;
		}

		m_numSamples = m_dataBytesRemaining / blockAlign;

		// 回到 data chunk 的起点，准备正式读取音频数据
		m_file.clear();
		m_file.seekg(m_dataStartPos, std::ios::beg);

		if (!m_file)
		{
			Close();
			return false;
		}

		return true;
	}
};
