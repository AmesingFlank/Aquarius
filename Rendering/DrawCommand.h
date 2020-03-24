#pragma once

#include <glm/glm.hpp>

enum class RenderMode:int {
	
	MultiphaseMesh = 0, 
	WaterMesh = 1,
	FlatMesh = 2,
	NormalMesh = 3,
	MirrorMesh = 4,

	RefractMesh = 5,
	Phase0Mesh = 6,
	Phase1Mesh = 7,

	Particles = 8,
	MAX = 9
};

enum class EnvironmentMode : int {
	Skybox = 0,
	CornellBox = 1,
	ChessBoard = 2,
	MAX = 3
};

inline bool isMultiphase(RenderMode mode) {
	return (int)mode <= 7 && (int)mode>=5;
}

inline bool isMeshMode(RenderMode mode) {
	return (int)mode <= 7;
}

struct DrawCommand {
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec3 cameraPosition;
	float windowWidth;
	float windowHeight;
	float FOV;
	float near;
	float far;

	RenderMode renderMode;
	
	bool simulationPaused;
	glm::vec3 lightPosition;

	EnvironmentMode environmentMode;
	unsigned int texSkybox;

	float containerSize;
	float cornellBoxSize;
};