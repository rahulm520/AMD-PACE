# Supporting DLRMv2 INT8 with AMD PACE

The standard [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization.html) can be used to quantize the DLRMv2 model. The quantized model can then be used with AMD PACE.

1. Load the FP32 model and data required for the model.
    ```python
    model = DLRM(...)
    model.load_state_dict(torch.load('/path/to/fp32/weights'))

    data = <iteratable data>
    ```

2. Define the Qconfig, an example is shown:
    ```python
    from torch.ao.quantization import (
        PerChannelMinMaxObserver,
        QConfig,
        HistogramObserver,
    )
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import float_qparams_weight_only_qconfig_4bit, QConfigMapping

    emb_qconfig = float_qparams_weight_only_qconfig_4bit
    global_static_qconfig = QConfig(
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_affine,
            dtype=torch.quint8,
            reduce_range=False,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    static_qconfig_mapping = (
        QConfigMapping()
        .set_global(global_static_qconfig)
        .set_module_name(
            "*.embedding_bags.*", emb_qconfig
        )
    )
    ```
    > Make sure to use the correct name for the embedding bag modules. EmbeddingBag operator does not support INT8 outputs, thus, special qconfig needs to be set.

3. Prepare and calibrate the model
    ```python
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    densex, index, offset, labels = data[0]
    model = prepare_fx(model, static_qconfig_mapping, example_inputs=(densex, index, offset, labels))
    for (densex, index, offset, labels) in data:
        model(densex, index, offset)
    model = convert_fx(model)
    ```

4. Convert to TorchScript and save the model
    ```python
    model.eval()
    model = torch.jit.trace(model, (densex, index, offset, labels), check_trace=True)
    model = torch.jit.freeze(model)
    torch.jit.save(model, '/path/to/dlrm_int8-pt_pace.pt')
    ```

5. The model saved in the previous step can be used with AMD PACE as:
    ```python
    import torch, pace
    pace.core.enable_pace_fusion(True)

    model = torch.jit.load('/path/to/dlrm_int8-pt_pace.pt')

    # Warmup for the model for the optimizations to take effect
    model(inputs)
    model(inputs)

    # Run the model for inference
    model(inputs)
    ```
