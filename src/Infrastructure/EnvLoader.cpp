#include "EnvLoader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

void EnvLoader::load(const std::string& path)
{
	std::cout << "\n[EnvLoader] Loading env from: " << path << std::endl;

	std::ifstream file(path);


	if (!file.is_open())
	{
		std::cout << "\n[EnvLoader] ERROR: Cannot open file!" << std::endl;
		return;
	}

	std::string line;
	int loadedCount = 0;

	while (std::getline(file, line))
	{
		if (line.empty() || line[0] == '#')
		{
			std::cout << "[EnvLoader] Skip line: " << line << std::endl;
			continue;
		}

		auto pos = line.find('=');
		if (pos == std::string::npos)
		{
			std::cout << "[EnvLoader] Invalid line: " << line << std::endl;
			continue;
		}

		std::string key = line.substr(0, pos);
		std::string value = line.substr(pos + 1);

		if (!value.empty() && value.front() == '"')
			value.erase(0, 1);

		if (!value.empty() && value.back() == '"')
			value.pop_back();

		_putenv_s(key.c_str(), value.c_str());

		std::cout << "[EnvLoader] Set: " << key << " = " << value << std::endl;

		loadedCount++;
	}

	std::cout << "[EnvLoader] Done. Loaded variables: " << loadedCount << std::endl;
}
