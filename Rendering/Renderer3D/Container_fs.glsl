#pragma once
#include<string>
static const std::string Container_fs = R"(

#version 330 core
in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
out vec4 FragColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

uniform vec3 cameraPosition;

void main()
{

    Material material;
    material.ambient=vec3(0.5);
    material.diffuse=vec3(0.5);

    material.specular=vec3(0.5);
    material.shininess=0.4;

    Light light;
    light.position= vec3(0,1000,0);
    light.ambient = vec3(0.2);
    light.diffuse = vec3(0.6);
    light.specular = vec3(0.01);

    light.constant=1;
    light.linear=0.005;
    light.quadratic=0.001;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 viewDir = normalize(cameraPosition - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128);

    vec3 halfwayDir = normalize(lightDir + viewDir);
    spec = pow(max(dot(norm, halfwayDir), 0.0), material.shininess*128);

    vec3 ambient  = light.ambient * material.ambient;
    vec3 diffuse  = light.diffuse * (diff * material.diffuse);
    vec3 specular = light.specular * (spec * material.specular);



    FragColor = vec4(ambient+diffuse+specular,1);
    float gamma = 2.2;
    FragColor.rgb = pow(FragColor.rgb, vec3(1.0/gamma));

}

)";
