# ver4.0 alpha_2.0: `reasoning_history`

`chat_template.jinja` takes a `reasoning_history` kwarg that decides which earlier-turn reasoning
(the harmony `analysis` channel) is rendered back into the prompt. The reasoning of the turn being
generated is not affected.

| Mode | Earlier-turn reasoning rendered |
|---|---|
| `none` (default) | Nothing (drop-all). Byte-identical to the template before this option existed. |
| `toolloop` | Reasoning of tool-call turns in the open tool loop; a later final answer closes the loop and drops it (the gpt-oss / harmony rule). |
| `keepall` | Reasoning of every earlier assistant turn, final answers included. |

Reasoning is read from `thinking`, else `reasoning_content`, else `reasoning`, so the messages that
OpenAI-compatible servers return work as they are. Any other value raises an error.

Serve a model with the mode it was trained with. The default is `none`, so a model trained with
`toolloop` must be given the kwarg explicitly.

## Switching modes

transformers:

```python
tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                        reasoning_history="toolloop")
```

vLLM, server-wide default:

```bash
vllm serve <model> --default-chat-template-kwargs '{"reasoning_history": "toolloop"}'
```

vLLM, per request (overrides the server default):

```json
{"model": "<model>", "messages": [...], "chat_template_kwargs": {"reasoning_history": "toolloop"}}
```

## Tests

```bash
python tests/test_reasoning_history.py
```
