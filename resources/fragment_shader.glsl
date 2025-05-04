#version 330 core

in vec2 fragmentTexCoord;

out vec4 color;

uniform sampler2D imageTexture;
uniform vec2 textureOffset;

void main() {
    color = texture(imageTexture, fragmentTexCoord + textureOffset);
}