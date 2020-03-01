/*

#include "FluidConfig.cuh"
#include "../Common/GpuCommons.h"

InitializationVolume getVolume(rapidjson::Value& v) {
	if (!v.HasMember("shape")) {
		throw std::string("no shape in volume in config file");
	}
	if (!v.HasMember("params")) {
		throw std::string("no params in volume");
	}
	if (!v.HasMember("phase")) {
		throw std::string("no phase in volume");
	}
	InitializationVolume result;
	rapidjson::Value& shape = v["shape"];
	rapidjson::Value& params = v["params"];
	rapidjson::Value& phase = v["phase"];
	
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

	if (!phase.IsInt()) {
		throw std::string("phase is not int");
	}
	result.phase = phase.GetInt();

	return result;

}


FluidConfig getConfigFromFile() {
	const std::string configPath = "./resources/LaunchConfigs.json";
	std::ifstream configFile;
	std::string configRawJSON;
	configFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	FluidConfig result;

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
		return result;
	}

	rapidjson::Document doc;
	doc.Parse(configRawJSON.c_str());



	rapidjson::Value& methodValue = doc["method"];
	if (!methodValue.IsString()) {
		std::cout << "ERROR:: method is not string" << std::endl;
		return result;
	}
	std::string method = methodValue.GetString();



	

	result.method = method;

	if (!doc.HasMember("NumPhases")) {
		std::cout << "ERROR:: missing NumPhases in config file" << std::endl;
		return result;
	}
	else {
		rapidjson::Value& numPhasesValue = doc["NumPhases"];
		if (!numPhasesValue.IsInt()) {
			std::cout << "ERROR:: numPhases is not int" << std::endl;
			return result;
		}
		result.phaseCount = numPhasesValue.GetInt();
	}

	if (!doc.HasMember("diffusion")) {
		std::cout << "ERROR:: missing diffusion in config file" << std::endl;
		return result;
	}
	else {
		rapidjson::Value& diffusionValue = doc["diffusion"];
		if (!diffusionValue.IsFloat()) {
			std::cout << "ERROR:: diffusion is not float" << std::endl;
			return result;
		}
		result.diffusionCoeff = diffusionValue.GetFloat();
	}

	if (!doc.HasMember("PhaseColors")) {
		std::cout << "ERROR:: missing PhaseColors in config file" << std::endl;
		return result;
	}
	else {
		rapidjson::Value& phaseColorsValue = doc["PhaseColors"];
		if (!phaseColorsValue.IsArray()) {
			std::cout << "ERROR:: PhaseColors is not an array" << std::endl;
			return result;
		}
		int phaseColorsLength = phaseColorsValue.Size();
		if (phaseColorsLength < result.phaseCount) {
			std::cout << "ERROR:: missing PhaseColors in config file" << std::endl;
			return result;
		}
		for (int i = 0; i < phaseColorsLength; ++i) {
			rapidjson::Value& thisColorValue = phaseColorsValue[i];
			if (!thisColorValue.IsArray() || thisColorValue.Size() < 4) {
				std::cout << "ERROR:: invalid color in phase "<<i
					<<"  not an arrar / not enough components (RGBA required)" << std::endl;
				return result;
			}

			float4 color;
			color.x = thisColorValue[0].GetFloat();
			color.y = thisColorValue[1].GetFloat();
			color.z = thisColorValue[2].GetFloat();
			color.w = thisColorValue[3].GetFloat();

			result.phaseColors.push_back(color);
		}


	}

	if (!doc.HasMember("sizeX") || !doc.HasMember("sizeY") || !doc.HasMember("sizeZ")) {
		std::cout << "ERROR:: missing size X/Y/Z in config file" << std::endl;
		return result;
	}

	rapidjson::Value& sizeX = doc["sizeX"];
	if (!sizeX.IsInt()) {
		std::cout << "ERROR:: sizeX is not int" << std::endl;
		return result;
	}
	result.sizeX = sizeX.GetInt();

	rapidjson::Value& sizeY = doc["sizeY"];
	if (!sizeY.IsInt()) {
		std::cout << "ERROR:: sizeY is not int" << std::endl;
		return result;
	}
	result.sizeY = sizeY.GetInt();

	rapidjson::Value& sizeZ = doc["sizeZ"];
	if (!sizeZ.IsInt()) {
		std::cout << "ERROR:: sizeZ is not int" << std::endl;
		return result;
	}
	result.sizeZ = sizeZ.GetInt();

	if (!doc.HasMember("volumes")) {
		std::cout << "ERROR:: missing volumes in config file" << std::endl;
		return result;
	}

	rapidjson::Value& volumes = doc["volumes"];
	if (!volumes.IsArray()) {
		std::cout << "ERROR::  volumes is not an array" << std::endl;
		return result;
	}
	for (int i = 0; i < volumes.Size(); ++i) {
		rapidjson::Value& v = volumes[i];
		try {
			result.initialVolumes.push_back(getVolume(v));
		}
		catch (std::string e) {
			std::cout << "ERROR:: " <<e<< std::endl;
			return result;
		}
	}

	return result;
	

	

}

*/