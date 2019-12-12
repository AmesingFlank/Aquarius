#include "FluidConfig.cuh"

InitializationVolume getVolume(rapidjson::Value& v) {
	if (!v.HasMember("shape")) {
		throw std::string("no shape in volume in config file");
	}
	if (!v.HasMember("params")) {
		throw std::string("no params in volume");
	}
	InitializationVolume result;
	rapidjson::Value& shape = v["shape"];
	rapidjson::Value& params = v["params"];
	
	if (!shape.IsString()) {
		throw std::string("shape is not string");
	}
	std::string shapeStr = shape.GetString();
	if (shapeStr == "sphere") {
		result.shapeType = ShapeType::Sphere;
	}
	else if (shapeStr == "square") {
		result.shapeType = ShapeType::Square;
	}
	else {
		throw std::string("invalid shape");
	}

	if (!params.IsArray()) {
		throw std::string(" params is not array");
	}
	for (int i = 0; i < params.Size(); ++i) {
		if (!params[i].IsFloat()) {
			throw std::string(" params[" + std::to_string(i) +"] is not float");
		}

		float p = params[i].GetFloat();
		result.params.push_back(p);
	}

	return result;

}


std::shared_ptr<FluidConfig> getConfig() {
	const std::string configPath = "./resources/LaunchConfigs.json";
	std::ifstream configFile;
	std::string configRawJSON;
	configFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{

		configFile.open(configPath);
		std::stringstream configStream;
		configStream << configFile.rdbuf();

		configFile.close();
		configRawJSON = configStream.str();

	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR:: could not read config file" << std::endl;
		return nullptr;
	}

	rapidjson::Document doc;
	doc.Parse(configRawJSON.c_str());


	if (!doc.HasMember("dimension") || !doc.HasMember("method")) {
		std::cout << "ERROR:: missing dimension/method in config file" << std::endl;
		return nullptr;
	}

	rapidjson::Value& dimensionValue = doc["dimension"];
	if (!dimensionValue.IsInt()) {
		std::cout << "ERROR:: dimension is not int" << std::endl;
		return nullptr;
	}
	int dimension = dimensionValue.GetInt();

	rapidjson::Value& methodValue = doc["method"];
	if (!methodValue.IsString()) {
		std::cout << "ERROR:: method is not string" << std::endl;
		return nullptr;
	}
	std::string method = methodValue.GetString();

	if (dimension == 3) {
		std::shared_ptr<FluidConfig3D> result = std::make_shared<FluidConfig3D>();
		result->dimension = dimension;
		result->method = method;

		if (!doc.HasMember("sizeX") || !doc.HasMember("sizeY") || !doc.HasMember("sizeZ")) {
			std::cout << "ERROR:: missing size X/Y/Z in config file" << std::endl;
			return nullptr;
		}

		rapidjson::Value& sizeX = doc["sizeX"];
		if (!sizeX.IsInt()) {
			std::cout << "ERROR:: sizeX is not int" << std::endl;
			return nullptr;
		}
		result->sizeX = sizeX.GetInt();

		rapidjson::Value& sizeY = doc["sizeY"];
		if (!sizeY.IsInt()) {
			std::cout << "ERROR:: sizeY is not int" << std::endl;
			return nullptr;
		}
		result->sizeY = sizeY.GetInt();

		rapidjson::Value& sizeZ = doc["sizeZ"];
		if (!sizeZ.IsInt()) {
			std::cout << "ERROR:: sizeZ is not int" << std::endl;
			return nullptr;
		}
		result->sizeZ = sizeZ.GetInt();

		if (!doc.HasMember("volumes")) {
			std::cout << "ERROR:: missing volumes in config file" << std::endl;
			return nullptr;
		}

		rapidjson::Value& volumes = doc["volumes"];
		if (!volumes.IsArray()) {
			std::cout << "ERROR::  volumes is not an array" << std::endl;
			return nullptr;
		}
		for (int i = 0; i < volumes.Size(); ++i) {
			rapidjson::Value& v = volumes[i];
			try {
				result->initialVolumes.push_back(getVolume(v));
			}
			catch (std::string e) {
				std::cout << "ERROR:: " <<e<< std::endl;
				return nullptr;
			}
		}

		return result;
	}

	else if (dimension == 2) {
		std::shared_ptr<FluidConfig2D> result = std::make_shared<FluidConfig2D>();
		result->dimension = dimension;
		result->method = method;

		if (!doc.HasMember("sizeX") || !doc.HasMember("sizeY")  ) {
			std::cout << "ERROR:: missing size X/Y in config file" << std::endl;
			return nullptr;
		}

		rapidjson::Value& sizeX = doc["sizeX"];
		if (!sizeX.IsInt()) {
			std::cout << "ERROR:: sizeX is not int" << std::endl;
			return nullptr;
		}
		result->sizeX = sizeX.GetInt();

		rapidjson::Value& sizeY = doc["sizeY"];
		if (!sizeY.IsInt()) {
			std::cout << "ERROR:: sizeY is not int" << std::endl;
			return nullptr;
		}
		result->sizeY = sizeY.GetInt();

		if (!doc.HasMember("volumes")) {
			std::cout << "ERROR:: missing volumes in config file" << std::endl;
			return nullptr;
		}

		rapidjson::Value& volumes = doc["volumes"];
		if (!volumes.IsArray()) {
			std::cout << "ERROR::  volumes is not an array" << std::endl;
			return nullptr;
		}
		for (int i = 0; i < volumes.Size(); ++i) {
			rapidjson::Value& v = volumes[i];
			try {
				result->initialVolumes.push_back(getVolume(v));
			}
			catch (std::string e) {
				std::cout << "ERROR:: " << e << std::endl;
				return nullptr;
			}
		}


		return result;
	}
	else {
		std::cout << "ERROR:: unsupported dimension value in config file" << std::endl;
		return nullptr;
	}


}