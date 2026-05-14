#!/usr/bin/env python3
"""
Pollinations 双模型图像生成 MCP Server
支持 Z-Image-Turbo (标准) 和 Qwen Image Plus (增强)
"""

import asyncio
import base64
import json
import os
import sys
from typing import Any
from urllib.parse import urlencode

import aiohttp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建 MCP Server 实例
server = Server("pollinations-image-gen", "1.0.0")

# Pollinations API 基础 URL
API_BASE = "https://image.pollinations.ai/prompt"

# 支持的模型列表
SUPPORTED_MODELS = {
    "zimage": {
        "name": "Z-Image-Turbo",
        "description": "标准模型，生成速度快，适合日常使用",
        "api_model": "zimage",
        "default_width": 1024,
        "default_height": 1024,
    },
    "qwen": {
        "name": "Qwen Image Plus",
        "description": "增强模型，文字渲染能力极强，适合海报、PPT、Logo、含中文文字的图像",
        "api_model": "qwen",  # Pollinations 中 Qwen Image Plus 的模型参数
        "default_width": 1024,
        "default_height": 1024,
    },
}

# 支持的分辨率列表
SUPPORTED_SIZES = [
    "1024x1024",  # 1:1
    "1280x720",   # 16:9
    "720x1280",   # 9:16
    "1024x768",   # 4:3
    "768x1024",   # 3:4
    "1440x720",   # 2:1 横屏壁纸
]

# 输出目录（可选，用于保存图片文件）
OUTPUT_DIR = os.getenv("POLLINATIONS_OUTPUT_DIR", "./pollinations-output")


async def generate_image_pollinations(
    prompt: str,
    model: str = "zimage",
    size: str = "1024x1024",
    seed: int = -1,
    enhance: bool = False,
    safe: bool = False,
    nologo: bool = True,
) -> dict[str, Any]:
    """调用 Pollinations API 生成图像"""
    
    # 验证模型
    if model not in SUPPORTED_MODELS:
        return {
            "success": False,
            "error": f"不支持的模型: {model}。支持: {', '.join(SUPPORTED_MODELS.keys())}"
        }
    
    # 验证分辨率
    if size not in SUPPORTED_SIZES:
        return {
            "success": False,
            "error": f"不支持的尺寸: {size}。支持: {', '.join(SUPPORTED_SIZES)}"
        }
    
    # 解析尺寸
    try:
        width, height = map(int, size.split("x"))
    except ValueError:
        return {
            "success": False,
            "error": f"尺寸格式错误: {size}。请使用 宽x高 格式，如 1024x1024"
        }
    
    # 构建请求参数
    api_model = SUPPORTED_MODELS[model]["api_model"]
    params = {
        "model": api_model,
        "width": width,
        "height": height,
        "nologo": str(nologo).lower(),
    }
    
    if seed != -1:
        params["seed"] = seed
    
    if enhance:
        params["enhance"] = "true"
    
    if safe:
        params["safe"] = "true"
    
    # 对 prompt 进行 URL 编码
    encoded_prompt = prompt
    
    # 构建完整 URL
    url = f"{API_BASE}/{encoded_prompt}?{urlencode(params)}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"API 调用失败: {response.status} - {error_text[:200]}"
                    }
                
                # 获取图像数据
                image_bytes = await response.read()
                
                if not image_bytes or len(image_bytes) < 100:
                    return {
                        "success": False,
                        "error": "生成的图像数据为空或太小"
                    }
                
                # 转换为 Base64
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # 可选：保存到本地文件
                saved_path = None
                if OUTPUT_DIR:
                    try:
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        import time
                        timestamp = int(time.time())
                        filename = f"pollinations_{model}_{timestamp}.png"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
                        saved_path = filepath
                    except Exception:
                        pass
                
                return {
                    "success": True,
                    "image_base64": image_base64,
                    "width": width,
                    "height": height,
                    "model": SUPPORTED_MODELS[model]["name"],
                    "seed": seed,
                    "saved_path": saved_path,
                }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "API 请求超时（60秒）"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"发生未知错误: {str(e)}"
            }


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """列出可用的 MCP 工具"""
    return [
        Tool(
            name="generate_pollinations_image_standard",
            description=f"""使用 Pollinations AI 的 Z-Image-Turbo（标准）模型生成图像。

**特点：**
- 生成速度快，适合日常使用
- 免费使用，无需 API Key
- 支持多种分辨率

**支持分辨率：**
{', '.join(SUPPORTED_SIZES)}

**使用场景：**
- 快速生成图像
- 日常创意设计
- 社交媒体配图
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图像描述提示词，建议使用英文以获得更好的效果"
                    },
                    "size": {
                        "type": "string",
                        "description": f"图像尺寸。默认 '1024x1024'。支持: {', '.join(SUPPORTED_SIZES)}",
                        "default": "1024x1024"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "随机种子，-1 表示随机。固定种子可复现相同结果",
                        "default": -1
                    },
                    "enhance": {
                        "type": "boolean",
                        "description": "是否启用 AI 提示词增强",
                        "default": False
                    },
                    "safe": {
                        "type": "boolean",
                        "description": "是否启用安全过滤",
                        "default": False
                    },
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="generate_pollinations_image_enhanced",
            description=f"""使用 Pollinations AI 的 Qwen Image Plus（增强）模型生成图像。

**特点：**
- 文字渲染能力极强，支持中文/英文文字
- 生成质量高，细节丰富
- 免费使用，无需 API Key
- 适合海报、PPT、Logo、图表等需要精确文字的场景

**支持分辨率：**
{', '.join(SUPPORTED_SIZES)}

**使用场景：**
- 中文/英文海报设计
- 带文字的Logo、商标
- PPT 插图、图表
- 电商产品图（含促销文字）
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图像描述提示词，需要包含文字时请用引号标注，如：'a poster with text \"会议通知\"'"
                    },
                    "size": {
                        "type": "string",
                        "description": f"图像尺寸。默认 '1024x1024'。支持: {', '.join(SUPPORTED_SIZES)}",
                        "default": "1024x1024"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "随机种子，-1 表示随机。固定种子可复现相同结果",
                        "default": -1
                    },
                    "enhance": {
                        "type": "boolean",
                        "description": "是否启用 AI 提示词增强",
                        "default": False
                    },
                    "safe": {
                        "type": "boolean",
                        "description": "是否启用安全过滤",
                        "default": False
                    },
                },
                "required": ["prompt"]
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Any) -> list[TextContent]:
    """调用工具"""
    
    if name == "generate_pollinations_image_standard":
        prompt = arguments.get("prompt", "")
        size = arguments.get("size", "1024x1024")
        seed = arguments.get("seed", -1)
        enhance = arguments.get("enhance", False)
        safe = arguments.get("safe", False)
        
        if not prompt:
            return [TextContent(
                type="text",
                text="错误: 缺少必需参数 'prompt'"
            )]
        
        result = await generate_image_pollinations(
            prompt=prompt,
            model="zimage",
            size=size,
            seed=seed,
            enhance=enhance,
            safe=safe,
        )
        
        return format_result(result, "zimage")
    
    elif name == "generate_pollinations_image_enhanced":
        prompt = arguments.get("prompt", "")
        size = arguments.get("size", "1024x1024")
        seed = arguments.get("seed", -1)
        enhance = arguments.get("enhance", False)
        safe = arguments.get("safe", False)
        
        if not prompt:
            return [TextContent(
                type="text",
                text="错误: 缺少必需参数 'prompt'"
            )]
        
        result = await generate_image_pollinations(
            prompt=prompt,
            model="qwen",
            size=size,
            seed=seed,
            enhance=enhance,
            safe=safe,
        )
        
        return format_result(result, "qwen")
    
    return [TextContent(
        type="text",
        text=f"未知工具: {name}"
    )]


def format_result(result: dict[str, Any], model_key: str) -> list[TextContent]:
    """格式化返回结果"""
    if result["success"]:
        model_name = SUPPORTED_MODELS[model_key]["name"]
        response_text = f"""✅ {model_name} 图像生成成功！

**配置信息：**
- 尺寸: {result['width']}x{result['height']}
- 模型: {result['model']}
- 随机种子: {result['seed']}

**Base64 数据（前100字符）：**
{result['image_base64'][:100]}...

**完整 Base64 数据：**
{result['image_base64']}
"""
        if result.get("saved_path"):
            response_text += f"\n**已保存至:** {result['saved_path']}"
        
        return [TextContent(
            type="text",
            text=response_text
        )]
    else:
        return [TextContent(
            type="text",
            text=f"❌ 图像生成失败: {result['error']}"
        )]


async def main():
    """启动 MCP Server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
