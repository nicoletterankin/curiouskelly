Shader "Kelly/RealisticSkin"
{
    Properties
    {
        _BaseMap ("Albedo", 2D) = "white" {}
        _BaseColor ("Color", Color) = (1,1,1,1)
        
        [Header(Subsurface Scattering)]
        _SSSColor ("SSS Color", Color) = (1, 0.3, 0.3, 1)
        _SSSIntensity ("SSS Intensity", Range(0, 1)) = 0.5
        _SSSDistortion ("SSS Distortion", Range(0, 1)) = 0.5
        _SSSPower ("SSS Power", Range(0.1, 10)) = 2
        _SSSScale ("SSS Scale", Range(0, 10)) = 2
        
        [Header(Surface)]
        _NormalMap ("Normal Map", 2D) = "bump" {}
        _NormalScale ("Normal Scale", Range(0, 2)) = 1
        _Smoothness ("Smoothness", Range(0, 1)) = 0.6
        _Metallic ("Metallic", Range(0, 1)) = 0
        
        [Header(Micro Details)]
        _MicroNormalMap ("Micro Normal", 2D) = "bump" {}
        _MicroNormalScale ("Micro Normal Scale", Range(0, 1)) = 0.3
        _MicroNormalTiling ("Micro Tiling", Float) = 20
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 300
        
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            
            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 tangentOS : TANGENT;
                float2 uv : TEXCOORD0;
            };
            
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float3 viewDirWS : TEXCOORD3;
                float3 tangentWS : TEXCOORD4;
                float3 bitangentWS : TEXCOORD5;
            };
            
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_NormalMap);
            SAMPLER(sampler_NormalMap);
            TEXTURE2D(_MicroNormalMap);
            SAMPLER(sampler_MicroNormalMap);
            
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseColor;
                float4 _SSSColor;
                float _SSSIntensity;
                float _SSSDistortion;
                float _SSSPower;
                float _SSSScale;
                float _NormalScale;
                float _Smoothness;
                float _Metallic;
                float _MicroNormalScale;
                float _MicroNormalTiling;
            CBUFFER_END
            
            Varyings vert(Attributes input)
            {
                Varyings output;
                
                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);
                
                output.positionCS = vertexInput.positionCS;
                output.positionWS = vertexInput.positionWS;
                output.normalWS = normalInput.normalWS;
                output.tangentWS = normalInput.tangentWS;
                output.bitangentWS = normalInput.bitangentWS;
                output.viewDirWS = GetWorldSpaceViewDir(vertexInput.positionWS);
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                
                return output;
            }
            
            // Subsurface Scattering approximation
            float3 SubsurfaceScattering(float3 viewDir, float3 normal, float3 lightDir)
            {
                // Translucency (light passing through)
                float3 H = normalize(lightDir + normal * _SSSDistortion);
                float VdotH = pow(saturate(dot(viewDir, -H)), _SSSPower) * _SSSScale;
                return _SSSColor.rgb * VdotH * _SSSIntensity;
            }
            
            float4 frag(Varyings input) : SV_Target
            {
                // Sample textures
                float4 albedo = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv) * _BaseColor;
                float3 normalTS = UnpackNormalScale(
                    SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, input.uv), 
                    _NormalScale
                );
                
                // Micro normals (tiled)
                float3 microNormalTS = UnpackNormalScale(
                    SAMPLE_TEXTURE2D(_MicroNormalMap, sampler_MicroNormalMap, input.uv * _MicroNormalTiling),
                    _MicroNormalScale
                );
                
                // Combine normals
                normalTS = BlendNormal(normalTS, microNormalTS);
                
                // Transform normal to world space
                float3 normalWS = TransformTangentToWorld(
                    normalTS,
                    half3x3(input.tangentWS, input.bitangentWS, input.normalWS)
                );
                normalWS = normalize(normalWS);
                
                float3 viewDirWS = normalize(input.viewDirWS);
                
                // Get main light
                Light mainLight = GetMainLight();
                float3 lightDir = mainLight.direction;
                float3 lightColor = mainLight.color;
                
                // Standard lighting
                float NdotL = saturate(dot(normalWS, lightDir));
                float3 diffuse = albedo.rgb * lightColor * NdotL;
                
                // Specular
                float3 halfDir = normalize(lightDir + viewDirWS);
                float NdotH = saturate(dot(normalWS, halfDir));
                float specular = pow(NdotH, _Smoothness * 100) * (1 - _Metallic);
                
                // Subsurface scattering
                float3 sss = SubsurfaceScattering(viewDirWS, normalWS, lightDir);
                
                // Combine
                float3 color = diffuse + specular * lightColor + sss * lightColor;
                
                // Ambient
                color += albedo.rgb * 0.1;
                
                return float4(color, 1);
            }
            ENDHLSL
        }
    }
    
    FallBack "Universal Render Pipeline/Lit"
}

