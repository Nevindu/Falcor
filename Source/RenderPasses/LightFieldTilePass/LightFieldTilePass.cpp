/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "LightFieldTilePass.h"

namespace
{
const char kShaderFile[] = "RenderPasses/LightFieldTilePass/LightFieldTilePass.cs.slang";

const char kMode[] = "mode";
const char kLFViewGridX[] = "lfViewGridX";
const char kLFViewGridY[] = "lfViewGridY";
const char kLFSourceViewX[] = "lfSourceViewX";
const char kLFSourceViewY[] = "lfSourceViewY";
const char kFixedOutputSize[] = "fixedOutputSize";

const char kSrcColor[] = "srcColor";
const char kSrcAlbedo[] = "srcAlbedo";
const char kSrcEmission[] = "srcEmission";
const char kSrcPosW[] = "srcPosW";
const char kSrcGuideNormalW[] = "srcGuideNormalW";
const char kSrcPNFwidth[] = "srcPNFwidth";
const char kSrcLinearZ[] = "srcLinearZ";
const char kSrcMvec[] = "srcMvec";

const char kColor[] = "color";
const char kAlbedo[] = "albedo";
const char kEmission[] = "emission";
const char kPosW[] = "posW";
const char kGuideNormalW[] = "guideNormalW";
const char kPNFwidth[] = "pnFwidth";
const char kLinearZ[] = "linearZ";
const char kMvec[] = "mvec";
const char kDstColor[] = "dstColor";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, LightFieldTilePass>();
}

LightFieldTilePass::LightFieldTilePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMode)
        {
            const std::string mode = value;
            if (mode == "Extract" || mode == "extract")
                mMode = Mode::Extract;
            else if (mode == "Insert" || mode == "insert")
                mMode = Mode::Insert;
            else
                logWarning("Unknown LightFieldTilePass mode '{}'. Using Extract.", mode);
        }
        else if (key == kLFViewGridX)
            mLFViewGrid.x = value;
        else if (key == kLFViewGridY)
            mLFViewGrid.y = value;
        else if (key == kLFSourceViewX)
            mLFSourceView.x = value;
        else if (key == kLFSourceViewY)
            mLFSourceView.y = value;
        else if (key == kFixedOutputSize)
            mFixedOutputSize = value;
        else
            logWarning("Unknown property '{}' in LightFieldTilePass properties.", key);
    }

    mLFViewGrid = max(mLFViewGrid, uint2(1));
    mLFSourceView = min(mLFSourceView, mLFViewGrid - uint2(1));

    const char* entryPoint = mMode == Mode::Extract ? "extractMain" : "insertMain";
    mpPass = ComputePass::create(mpDevice, kShaderFile, entryPoint);
}

Properties LightFieldTilePass::getProperties() const
{
    Properties props;
    props[kMode] = mMode == Mode::Extract ? "Extract" : "Insert";
    props[kLFViewGridX] = mLFViewGrid.x;
    props[kLFViewGridY] = mLFViewGrid.y;
    props[kLFSourceViewX] = mLFSourceView.x;
    props[kLFSourceViewY] = mLFSourceView.y;
    props[kFixedOutputSize] = mFixedOutputSize;
    return props;
}

uint2 LightFieldTilePass::getTileSize(uint2 atlasSize) const
{
    return uint2(std::max(1u, atlasSize.x / mLFViewGrid.x), std::max(1u, atlasSize.y / mLFViewGrid.y));
}

uint2 LightFieldTilePass::getOutputSize(const CompileData& compileData) const
{
    if (mMode == Mode::Insert)
        return compileData.defaultTexDims;
    if (mFixedOutputSize.x > 0 && mFixedOutputSize.y > 0)
        return mFixedOutputSize;
    return getTileSize(compileData.defaultTexDims);
}

RenderPassReflection LightFieldTilePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 outputSize = getOutputSize(compileData);

    if (mMode == Mode::Extract)
    {
        reflector.addInput(kSrcColor, "Atlas source color").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcAlbedo, "Atlas source albedo").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcEmission, "Atlas source emission").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcPosW, "Atlas source world position").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcGuideNormalW, "Atlas source guide normal").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcPNFwidth, "Atlas source position/normal fwidth").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcLinearZ, "Atlas source linear depth").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addInput(kSrcMvec, "Atlas source motion vectors").bindFlags(ResourceBindFlags::ShaderResource);

        const auto outputFlags = ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource;
        reflector.addOutput(kColor, "Source-view color").bindFlags(outputFlags).format(ResourceFormat::RGBA32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kAlbedo, "Source-view albedo").bindFlags(outputFlags).format(ResourceFormat::RGBA32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kEmission, "Source-view emission").bindFlags(outputFlags).format(ResourceFormat::RGBA32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kPosW, "Source-view world position").bindFlags(outputFlags).format(ResourceFormat::RGBA32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kGuideNormalW, "Source-view guide normal").bindFlags(outputFlags).format(ResourceFormat::RGBA32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kPNFwidth, "Source-view position/normal fwidth").bindFlags(outputFlags).format(ResourceFormat::RG32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kLinearZ, "Source-view linear depth").bindFlags(outputFlags).format(ResourceFormat::RG32Float).texture2D(outputSize.x, outputSize.y);
        reflector.addOutput(kMvec, "Source-view motion vectors").bindFlags(outputFlags).format(ResourceFormat::RG32Float).texture2D(outputSize.x, outputSize.y);
    }
    else
    {
        reflector.addInput(kSrcColor, "Denoised source-view color").bindFlags(ResourceBindFlags::ShaderResource);
        reflector.addOutput(kDstColor, "Atlas color with denoised source tile")
            .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(outputSize.x, outputSize.y);
    }

    return reflector;
}

void LightFieldTilePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpPass->getRootVar();
    auto cb = var["CB"];

    ref<Texture> pOutput = mMode == Mode::Extract ? renderData.getTexture(kColor) : renderData.getTexture(kDstColor);
    if (!pOutput)
    {
        logWarning("LightFieldTilePass::execute() - missing output resource.");
        return;
    }

    const uint2 dispatchSize = uint2(pOutput->getWidth(), pOutput->getHeight());
    const ref<Texture> pAtlas = mMode == Mode::Extract ? renderData.getTexture(kSrcColor) : pOutput;
    const uint2 atlasSize = uint2(pAtlas->getWidth(), pAtlas->getHeight());
    const uint2 tileSize = mMode == Mode::Extract ? dispatchSize : getTileSize(atlasSize);

    cb["gAtlasDim"] = atlasSize;
    cb["gTileDim"] = tileSize;
    cb["gViewGrid"] = mLFViewGrid;
    cb["gSourceView"] = mLFSourceView;

    if (mMode == Mode::Extract)
    {
        var["gSrcColor"] = renderData.getTexture(kSrcColor);
        var["gSrcAlbedo"] = renderData.getTexture(kSrcAlbedo);
        var["gSrcEmission"] = renderData.getTexture(kSrcEmission);
        var["gSrcPosW"] = renderData.getTexture(kSrcPosW);
        var["gSrcGuideNormalW"] = renderData.getTexture(kSrcGuideNormalW);
        var["gSrcPNFwidth"] = renderData.getTexture(kSrcPNFwidth);
        var["gSrcLinearZ"] = renderData.getTexture(kSrcLinearZ);
        var["gSrcMvec"] = renderData.getTexture(kSrcMvec);

        var["gColor"] = renderData.getTexture(kColor);
        var["gAlbedo"] = renderData.getTexture(kAlbedo);
        var["gEmission"] = renderData.getTexture(kEmission);
        var["gPosW"] = renderData.getTexture(kPosW);
        var["gGuideNormalW"] = renderData.getTexture(kGuideNormalW);
        var["gPNFwidth"] = renderData.getTexture(kPNFwidth);
        var["gLinearZ"] = renderData.getTexture(kLinearZ);
        var["gMvec"] = renderData.getTexture(kMvec);
    }
    else
    {
        var["gSrcColor"] = renderData.getTexture(kSrcColor);
        var["gDstColor"] = renderData.getTexture(kDstColor);
    }

    mpPass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}
