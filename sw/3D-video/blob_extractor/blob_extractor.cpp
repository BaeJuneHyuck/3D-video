#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <deque>
using namespace std;

// -- < config below > --

const float frame_time = 1000.0f / 15.0f;

// -- < config above > --
struct chunk_header
{
	int64_t timestamp;
	int64_t len;
};
struct chunk
{
	chunk_header header;
	array<char, 4 * 1024 * 1024> image;
};

deque<chunk> chunks;
chunk c;

int main(int argc, char* argv[])
{
	if (argc != 2 && argc != 3)
	{
		cout << "Usage: " << argv[0] << " cam_count" << endl;
		return 0;
	}

	const int cam_count = stoi(argv[1]);
	cout << "Cam count = " << cam_count << endl;
	for (int cam_index = 0; cam_index < cam_count; cam_index++)
	{
		cout << "\tProcessing cam #" << cam_index << endl;

		chunks.clear();

		ifstream f(string("jpg_blob") + to_string(cam_index), ios::binary);

		if (!f.is_open())
			return 0;

		for (int i = 0; i < 3; i++)
		{
			f.read((char*)& c.header.timestamp, sizeof(int64_t));
			f.read((char*)& c.header.len, sizeof(int64_t));

			if (!f.good())
				break;

			chunks.push_back(c);
			f.read(&chunks[chunks.size() - 1].image[0], c.header.len);
		}

		if (argc == 3)
		{
			ofstream f(string("sample") + to_string(cam_index) + ".jpg", ios::binary);
			f.write(&chunks[0].image[0], chunks[0].header.len);

			continue;
		}

		for (int i = 0;; i++)
		{
			float t = i * frame_time;

			float t0 = abs(t - chunks[0].header.timestamp / 1000.0f);
			float t1 = abs(t - chunks[1].header.timestamp / 1000.0f);
			float t2 = abs(t - chunks[2].header.timestamp / 1000.0f);

			if (t0 < t1)
			{
				cout << "t = " << t << ", t0 = " << t0 << ", t1 = " << t1 << ", t2 = " << t2 << endl;
				cout << " - too slow sampling: copying previous image..." << endl;
				ofstream f(string("view_") + to_string(i * cam_count + cam_index) + ".jpg", ios::binary);
				f.write(&chunks[0].image[0], chunks[0].header.len);
				continue;
			}
			else if (t1 < t2)
			{
				ofstream f(string("view_") + to_string(i * cam_count + cam_index) + ".jpg", ios::binary);
				f.write(&chunks[1].image[0], chunks[1].header.len);
			}
			else
			{
				cout << "t = " << t << ", t0 = " << t0 << ", t1 = " << t1 << ", t2 = " << t2 << endl;
				cout << " - too fast sampling: skipping image..." << endl;
				i--;
			}

			chunks.pop_front();

			f.read((char*)& c.header.timestamp, sizeof(int64_t));
			f.read((char*)& c.header.len, sizeof(int64_t));

			if (!f.good())
				break;

			chunks.push_back(c);
			f.read(&chunks[chunks.size() - 1].image[0], c.header.len);
		}
	}
}
