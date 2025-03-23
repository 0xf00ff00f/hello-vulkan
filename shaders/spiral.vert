#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec3 vertexPosition;
layout(location = 1) out vec3 vertexNormal;
layout(location = 2) out vec2 vertexTexCoord;

layout(set = 0, binding = 0) uniform Entity
{
    mat4 mvp;
    mat4 modelMatrix;
    vec4 texCoordOffset;
    vec4 globalLight;
}
entity;

void main(void)
{
    vertexPosition = vec3(entity.modelMatrix * vec4(position, 1.0));
    vertexNormal = normalize(mat3(entity.modelMatrix) * normal);
    vertexTexCoord = texCoord;
    gl_Position = entity.mvp * vec4(position, 1.0);
}
