#pragma once


class WindowInfo
{
public:
	static WindowInfo& instance()
	{
		static WindowInfo    instance; // Guaranteed to be destroyed.
		return instance;
	}

	int windowHeight;
	int windowWidth;




public:
	WindowInfo(WindowInfo const&) = delete;
	void operator=(WindowInfo const&) = delete;

private:
	WindowInfo() {
		

	}
};
