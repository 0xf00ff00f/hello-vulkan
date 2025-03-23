#version 450

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexTexCoord;

layout(set = 0, binding = 0) uniform Entity
{
    mat4 mvp;
    mat4 modelMatrix;
    vec4 texCoordOffset;
    vec4 globalLight;
}
entity;

layout(location = 0) out vec4 fragColor;

const float ambient = 0.15;
const vec3 globalLight = vec3(0, -5, 0);

float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float pattern(vec2 id, vec2 p, vec2 offs)
{
    id = mod(id + offs, 16);
    float r = 0.5 * random(id);
    vec2 o = vec2(random(id + vec2(0, 1)), random(id + vec2(1, 0))) + offs;
    float d = distance(p, o);
    return 1.0 - smoothstep(r - .05, r, d);
}

void main(void)
{
    vec2 texCoord = vec2(vertexTexCoord.x * 4.0, vertexTexCoord.y);
    texCoord += entity.texCoordOffset.xy;
    texCoord *= 20.0;

    vec2 p = fract(texCoord) - 0.5;
    vec2 id = floor(texCoord);

    float l = 0;
    for (float i = -1; i <= 1; ++i)
    {
        for (float j = -1; j <= 1; ++j)
        {
            l += pattern(id, p, vec2(i, j));
        }
    }
    l = clamp(l, 0, 1);

    vec3 color = vec3(l);

    float v = ambient + max(dot(vertexNormal, normalize(entity.globalLight.xyz - vertexPosition)), 0.0);
    fragColor = vec4(v * color, 0.4);
}
