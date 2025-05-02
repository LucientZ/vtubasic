#version 330 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec2 vertexTexCoord;

uniform mat3 modelView;

out vec2 fragmentTexCoord;

void main() {
    vec3 transformedPos = modelView * vec3(vertexPos.xy, 1.0); // 1.0 is the homogeneous coordinate
    gl_Position = vec4(transformedPos.xy, vertexPos.z, 1.0);
    fragmentTexCoord = vertexTexCoord;
}