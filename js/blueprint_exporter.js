export function generateSubgraphFromSuggestions(nodeClassTypes, widgetValues = {}) {
    const KNOWN_TYPES = {
        "CheckpointLoaderSimple": {
            defaultWidgets: ["sd_xl_base_1.0.safetensors"],
            defaultNamed: { ckpt_name: "sd_xl_base_1.0.safetensors" },
            outputs: ["MODEL", "CLIP", "VAE"],
        },
        "CLIPTextEncode": {
            defaultWidgets: ["masterpiece, best quality"],
            defaultNamed: { text: "masterpiece, best quality" },
            outputs: ["CONDITIONING"],
        },
        "EmptyLatentImage": {
            defaultWidgets: [1024, 1024, 1],
            defaultNamed: { width: 1024, height: 1024, batch_size: 1 },
            outputs: ["LATENT"],
        },
        "KSampler": {
            defaultWidgets: [0, "fixed", 20, 7.0, "dpmpp_2m", "karras", 1.0],
            defaultNamed: {
                seed: 0,
                steps: 20,
                cfg: 7.0,
                sampler_name: "dpmpp_2m",
                scheduler: "karras",
                denoise: 1.0,
            },
            outputs: ["LATENT"],
        },
        "VAEDecode": {
            outputs: ["IMAGE"],
        },
        "SaveImage": {
            defaultWidgets: ["ComfyUI"],
            defaultNamed: { filename_prefix: "ComfyUI" },
            outputs: ["IMAGE"],
        },
        "LoadImage": {
            defaultWidgets: ["example.png"],
            defaultNamed: { image: "example.png" },
            outputs: ["IMAGE"],
        },
        "ControlNetLoader": {
            defaultWidgets: ["control_v11p_sd15_depth.pth"],
            defaultNamed: { control_net_name: "control_v11p_sd15_depth.pth" },
            outputs: ["CONTROL_NET"],
        },
        "ControlNetApply": {
            defaultWidgets: [1.0],
            defaultNamed: { strength: 1.0 },
            outputs: ["CONDITIONING"],
        },
        "LoraLoader": {
            defaultWidgets: ["lora_one.safetensors", 1.0, 1.0],
            defaultNamed: {
                lora_name: "lora_one.safetensors",
                strength_model: 1.0,
                strength_clip: 1.0,
            },
            outputs: ["MODEL", "CLIP"],
        },
        "FaceDetailer": {
            defaultWidgets: [384, true, 1024, 0, "fixed", 20, 7.0, "dpmpp_2m", "karras", 0.5],
            defaultNamed: {
                guide_size: 384,
                guide_size_for: true,
                max_size: 1024,
                seed: 0,
                steps: 20,
                cfg: 7.0,
                sampler_name: "dpmpp_2m",
                scheduler: "karras",
                denoise: 0.5,
            },
            outputs: ["IMAGE"],
        },
        "LatentUpscale": {
            defaultWidgets: ["nearest-exact", 1536, 1536, "disabled"],
            defaultNamed: {
                upscale_method: "nearest-exact",
                width: 1536,
                height: 1536,
                crop: "disabled",
            },
            outputs: ["LATENT"],
        },
        "SaveAnimatedWEBP": {
            defaultWidgets: ["ComfyUI", 8.0, true, 80, "default"],
            defaultNamed: {
                filename_prefix: "ComfyUI",
                fps: 8.0,
                lossless: true,
                quality: 80,
                method: "default",
            },
            outputs: ["IMAGE"],
        },
        "UNETLoader": {
            defaultWidgets: ["flux1-dev-Q4_0.gguf", "default"],
            defaultNamed: { unet_name: "flux1-dev-Q4_0.gguf", weight_dtype: "default" },
            outputs: ["MODEL"],
        },
        "DualCLIPLoader": {
            defaultWidgets: ["clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "flux"],
            defaultNamed: {
                clip_name1: "clip_l.safetensors",
                clip_name2: "t5xxl_fp8_e4m3fn.safetensors",
                type: "flux",
            },
            outputs: ["CLIP"],
        },
        "VAELoader": {
            defaultWidgets: ["ae.safetensors"],
            defaultNamed: { vae_name: "ae.safetensors" },
            outputs: ["VAE"],
        },
        "VAEEncode": {
            outputs: ["LATENT"],
        },
    };

    // Validation: check each class_type against known types
    for (const type of nodeClassTypes) {
        if (!KNOWN_TYPES[type]) {
            console.warn(`[comfy_agent] unknown node type: ${type}`);
            return null;
        }
    }

    // Build nodes with positions and default widget values
    const nodes = [];
    let x = 0;

    for (let i = 0; i < nodeClassTypes.length; i++) {
        const type = nodeClassTypes[i];
        const typeDef = KNOWN_TYPES[type];
        const nodeId = i + 1;

        let widgets_values = [...(typeDef.defaultWidgets || [])];
        let widgets_named = { ...(typeDef.defaultNamed || {}) };

        // Apply widget overrides if provided for this type
        const widgetOverride = widgetValues[type];
        if (widgetOverride) {
            if (Array.isArray(widgetOverride.widgets_values)) {
                widgets_values = [...widgetOverride.widgets_values];
            }
            if (widgetOverride.widgets_named && typeof widgetOverride.widgets_named === "object") {
                widgets_named = { ...widgets_named, ...widgetOverride.widgets_named };
            }
        }

        const nodeDef = {
            id: nodeId,
            type: type,
            pos: [x + i * 20, 0],
            size: typeDef.size || [320, 100],
            flags: {},
            order: i,
            mode: 0,
            outputs: (typeDef.outputs || []).map((output) => ({
                name: output,
                type: output,
                links: [],
            })),
            properties: { "Node name for S&R": type },
            widgets_values,
            widgets_named,
        };

        nodes.push(nodeDef);
    }

    // Create links between consecutive nodes based on type compatibility
    const links = [];

    for (let i = 0; i < nodeClassTypes.length - 1; i++) {
        const fromType = nodeClassTypes[i];
        const toType = nodeClassTypes[i + 1];
        const fromDef = KNOWN_TYPES[fromType];
        const toDef = KNOWN_TYPES[toType];

        let fromOutput = 0;
        let toInput = 0;

        // Map output type to output slot on from node
        if (fromDef && fromDef.outputs) {
            fromOutput = fromDef.outputs.findIndex(
                (o) => o === "MODEL" || o === "CONDITIONING" || o === "LATENT" || o === "IMAGE" || o === "VAE" || o === "CONTROL_NET"
            );
            if (fromOutput < 0) fromOutput = 0;
        }

        // Map input type to input slot on to node
        if (toDef && toDef.outputs) {
            // Inputs in LiteGraph/ComfyUI are defined per node type
            // Use a simple heuristic: first output of from -> first input of to
            toInput = 0;
        }

        links.push([fromDef.id || (i + 1), fromOutput, toDef.id || (i + 2), toInput]);
    }

    // Post-process links to match the 6-element format: [from_node, from_slot, to_node, to_slot, , type]
    const processedLinks = links.map((link) => [
        link[0],
        link[1],
        link[2],
        link[3],
        0,
        "AUTO",
    ]);

    const extra = {
        comfy_agent_blueprint: {
            name: "generated",
            trigger_keywords: nodeClassTypes,
            note: "Generated by comfy_agent P1.4 Selective Node Export to Blueprint",
        },
    };

    return {
        last_node_id: nodeClassTypes.length,
        last_link_id: processedLinks.length,
        nodes,
        links: processedLinks,
        groups: [],
        config: {},
        extra,
        version: 0.4,
    };
}