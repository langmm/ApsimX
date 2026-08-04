import os
import json
import pprint
import argparse
import requests
import warnings
from pydantic import BaseModel
from config import ModelInput, InteractiveModelInput


_n8n_dir = os.path.abspath(os.path.dirname(__file__))
_scratch_dir = os.path.join(_n8n_dir, "scratch")
_n8n_address = "https://tools.uiuc.chat/api/v1"
_name2cls = {
    "start": ModelInput,
    "start-interactive": InteractiveModelInput,
}
_type_mapping = {
    "string": "text",
    "integer": "number",
    "number": "number",
    "boolean": "checkbox",
}
_n8n_type_nodefault = ["password", "html", "hiddenField", "file"]
_n8n_type_noplaceholder = [
    "dropdown", "checkbox", "radio", "date", "file"
]
_zero_indicates_null = ["year", "timestep"]


def tool_update_matches(updates, existing, remove_match=False):
    if not isinstance(existing, type(updates)):
        return False
    if isinstance(updates, list):
        if len(updates) != len(existing):
            return False
        out = True
        for v1, v2 in zip(updates, existing):
            if not tool_update_matches(v1, v2):
                out = False
        return out
    elif isinstance(updates, dict):
        out = True
        for k in list(updates.keys()):
            iout = True
            if k not in existing:
                iout = False
            elif not tool_update_matches(updates[k], existing[k]):
                iout = False
            if remove_match and iout:
                del updates[k]
            if not iout:
                out = False
        return out
    return (updates == existing)


def dump_to_scratch(payload: dict, output: str | bool | None,
                    default: str):
    if not output:
        return
    if output is True:
        output = os.path.join(_scratch_dir, f"{default}.json")
    else:
        if not os.path.splitext(output)[-1]:
            output += ".json"
        if not os.path.isabs(output):
            output = os.path.join(_scratch_dir, output)
    with open(output, 'w') as fd:
        json.dump(payload, fd, indent=2)
    print(f"{default} written to \"{output}\"")


def jsonschema_to_n8n(field_name: str, details: dict):
    if "type" not in details:
        assert "anyOf" in details
        anyOf = details.pop("anyOf")
        if {'type': 'null'} in anyOf:
            anyOf.remove({'type': 'null'})
        if {'format': 'duration', 'type': 'string'} in anyOf:
            anyOf.remove({'format': 'duration', 'type': 'string'})
        if (({'format': 'date-time', 'type': 'string'} in anyOf
             and {'type': 'string'} in anyOf)):
            anyOf.remove({'type': 'string'})
        if len(anyOf) == 1:
            details.update(anyOf[0])
        else:
            assert {'type': 'string'} in anyOf
        # import pdb; pdb.set_trace()
    pydantic_type = details.get("type", "string")
    pydantic_enum = details.get("enum", None)
    if isinstance(pydantic_type, list):
        if 'null' in pydantic_type:
            pydantic_type.remove('null')
        assert len(pydantic_type) == 1
        pydantic_type = pydantic_type[0]
    if ((pydantic_type == "string"
         and details.get("format", None) == "date-time")):
        n8n_type = "date"
    elif pydantic_type == "array" and "enum" in details.get("items", {}):
        n8n_type = _type_mapping.get(details["items"]["type"], "text")
        pydantic_enum = details["items"]["enum"]
    else:
        n8n_type = _type_mapping.get(pydantic_type, "text")

    # Special override for email strings
    if field_name == "email" or details.get("format") == "email":
        n8n_type = "email"
    elif pydantic_enum:
        if pydantic_type == "array":
            n8n_type = "checkbox"
        else:
            n8n_type = "radio"

    field_def = {
        "fieldLabel": field_name,
        "fieldName": field_name.replace("_", " ").title(),
        "fieldType": n8n_type,
    }
    if n8n_type == "date":
        field_def["formatDate"] = "YYYY-MM-DD"
    if ((details.get("description", None) is not None
         and n8n_type not in _n8n_type_noplaceholder)):
        field_def["placeholder"] = details["description"]
    if details.get("default", None) is not None:
        field_default = details["default"]
        if isinstance(field_default, list):
            if n8n_type not in ["checkbox"]:
                field_default = ",".join(field_default)
        elif isinstance(field_default, dict):
            field_default = json.dumps(field_default)
        if n8n_type not in _n8n_type_noplaceholder:
            field_def.setdefault("placeholder", "")
            field_def["placeholder"] += f" (e.g. {field_default})"
        if n8n_type not in _n8n_type_nodefault:
            field_def["defaultValue"] = field_default
    if pydantic_enum:
        field_def["fieldOptions"] = {
            "values": [{"option": k} for k in pydantic_enum]
        }
    if ((n8n_type == "number" and "defaultValue" not in field_def
         and field_name not in _zero_indicates_null)):
        raise RuntimeError(f"Default must be defined for number fields "
                           f"to prevent the n8n form from autofilling "
                           f"with 0 (field = \"{field_name}\")")
    return field_def


def pydantic_to_n8n_fields(model: type[BaseModel]):
    schema = model.model_json_schema(union_format="primitive_type_array")
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    form_fields = []
    for field_name, details in properties.items():
        field_def = jsonschema_to_n8n(field_name, details)
        field_def["requiredField"] = field_name in required_fields
        form_fields.append(field_def)
    return form_fields


def pydantic_to_n8n_form(name: str, model: type[BaseModel]):
    fields = pydantic_to_n8n_fields(model)
    out = {
        "name": f"ApsimX {name} Form Trigger",
        "type": "n8n-nodes-base.formTrigger",
        "typeVersion": 2.1,
        "parameters": {
            "path": f"apsimx-{name}-form".lower(),
            "formTitle": f"Run ApsimX {name}",
            "formFields": {
                "values": fields,
                "options": {},
            },
        },
        "position": [600, 380],
    }
    return out


def name_to_n8n_form(name: str, output: str | bool | None = None):
    model = _name2cls[name]
    form = pydantic_to_n8n_form(name, model)
    dump_to_scratch(form, output, f"n8n-form-apsimx-{name}")
    return form


def n8n_api_request(path: str, action: str,
                    headers: dict | None = None,
                    verbose: bool = False,
                    dry_run: bool = False,
                    **kwargs):
    if headers is None:
        headers = {}
    credentials = headers.get(
        'X_N8N_API_KEY', os.environ.get('X_N8N_API_KEY', None))
    if credentials is None:
        raise RuntimeError("No credentials provided and "
                           "\"X_N8N_API_KEY\" environment "
                           "variable not set")
    headers['X-N8N-API-KEY'] = credentials
    if dry_run:
        print(action.upper(), f'{_n8n_address}/{path}', '[DRY RUN]')
        return
    r = getattr(requests, action)(
        f'{_n8n_address}/{path}',
        headers=headers,
        **kwargs
    )
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print(r.content)
        raise
    out = r.json()
    if verbose:
        print(action.upper(), f'{_n8n_address}/{path}')
        pprint.pprint(out)
    return out


def query_n8n_service(name: str, toolname: str | None = None,
                      output: str | bool | None = None,
                      allow_multiple: bool = False,
                      required: bool = False, **kwargs):
    if toolname is None:
        toolname = f'ApsimX {name} Tool'
    response = n8n_api_request(
        'workflows', 'get', params={'name': toolname}, **kwargs
    )
    if required and len(response['data']) == 0:
        raise RuntimeError(f"No tool found matching name \"{toolname}\"")
    if len(response['data']) > 0:
        dump_to_scratch(response, output, toolname.replace(' ', '_'))
    elif len(response['data']) > 1 and (not allow_multiple):
        raise RuntimeError(
            f"More than one tool matching name \"{toolname}\":\n"
            f"{pprint.pformat(response['data'])}")
    return response


def remove_n8n_service(name: str, toolname: str | None = None,
                       output: str | bool | None = None,
                       dry_run: bool = False, idstr: str | None = None,
                       **kwargs):
    if toolname is None:
        toolname = f'ApsimX {name} Tool'
    if not toolname.startswith("ApsimX"):
        raise RuntimeError(
            f"Cannot remove someone else's tool: \"{toolname}\"")
    if output is True:
        output = toolname.replace(' ', '_') + "_PREV"
    if idstr is None or output:
        response = query_n8n_service(
            name, toolname=toolname,
            output=output, required=True,
            **kwargs)
        if idstr is None:
            idstr = response['data'][0]["id"]
        else:
            assert response['data'][0]["id"] == idstr
    return n8n_api_request(
        f'workflows/{idstr}', 'delete', dry_run=dry_run, **kwargs
    )


def publish_n8n_service(name: str,
                        service_address: str | None = None,
                        toolname: str | None = None,
                        output_request: str | bool | None = None,
                        output_tool: str | bool | None = None,
                        output_form: str | bool | None = None,
                        output_prev: str | bool | None = None,
                        overwrite: bool = False,
                        update: bool | str = False,
                        dry_run: bool = False,
                        **kwargs):
    if toolname is None:
        toolname = f'ApsimX {name} Tool'
    # Check if the workflow exists
    if output_tool and not output_prev:
        output_prev = True
    if output_prev is True:
        output_prev = toolname.replace(' ', '_') + "_PREV"
    response = query_n8n_service(
        name, toolname=toolname,
        output=output_prev,
        required=(update == "required"),
        **kwargs)
    existing = None
    if len(response['data']) > 0:
        if not (overwrite or update):
            dump_to_scratch(response["data"], output_tool,
                            toolname.replace(' ', '_'))
            warnings.warn(
                f"A tool already exists with name \"{toolname}\":\n"
                f"{pprint.pformat(response['data'])}")
            return
        existing = response['data'][0]
        if not service_address:
            # Use existing address
            service_address = (
                existing["nodes"][-1]["parameters"]["url"].rsplit(
                    "/", 1)[0]
            )
    if not (service_address or output_form):
        output_form = True
    form_node = name_to_n8n_form(name, output=output_form)
    if not service_address:
        warnings.warn("Cannot create n8n tool without service address")
        return
    transform_node = {
        "name": "Strip empty fields",
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "parameters": {
            "jsCode": "\n".join([
                "// Remove empty fields",
                "for (const key in $json) {",
                "  if ($json[key] === null || $json[key] === \"\" || "
                "$json[key] === undefined) {",
                "    delete $json[key];",
                "  }",
                "}",
            ] + [
                "if ($json." + x + " == 0) { delete $json." + x + "; }"
                for x in _zero_indicates_null
            ] + [
                "return $json;"
            ])
        }
    }
    request_node = {
        "name": "HTTP Request",
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "parameters": {
            "method": "POST",
            "url": f"{service_address.rstrip('/')}/{name}",
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{$json.toJsonString()}}",
            "options": {
                "response": {
                    "response": {
                        "responseFormat": "json"
                    }
                }
            },
        },
    }
    nodes = [
        form_node,
        transform_node,
        request_node,
    ]
    for i, x in enumerate(nodes):
        x["position"] = [
            400 + i * 200,
            400
        ]
    request = {
        "name": toolname,
        "nodes": nodes,
        "connections": {
            form_node["name"]: {
                "main": [[{
                    "node": transform_node["name"],
                    "type": "main",
                    "index": 0,
                }]],
            },
            transform_node["name"]: {
                "main": [[{
                    "node": request_node["name"],
                    "type": "main",
                    "index": 0,
                }]],
            },
        },
        "settings": {
            # "saveExecutionProgress": False,
            "saveManualExecutions": False,
            # "saveDataErrorExecution": "none",
            # "saveDataSuccessExecution": "none",
            # "executionTimeout": 3600,
            # "errorWorkflow": "",
            # "timezone": "America/New_York",
            "executionOrder": "v1",
        },
        "projectId": "dIzmFYmKV9uZMjnZ",
    }
    if existing:
        updates = request.copy()
        updates.pop("projectId", None)
        tool_update_matches(updates, existing,
                            remove_match=True)
        if not updates:
            print("No updates required")
            return
    if overwrite and existing:
        remove_n8n_service(
            name, toolname=toolname, dry_run=dry_run,
            idstr=existing["id"], **kwargs
        )
        update = False
    if update and existing:
        request["id"] = existing["id"]
        dump_to_scratch(request, output_request,
                        f"{name}-tool-request-update")
        response = n8n_api_request(
            f'workflows/{request["id"]}', 'put',
            json=request, dry_run=dry_run,
            headers={'accept': 'application/json'}, **kwargs)
    else:
        dump_to_scratch(request, output_request,
                        f"{name}-tool-request")
        response = n8n_api_request(
            'workflows', 'post', json=request, dry_run=dry_run,
            headers={'accept': 'application/json'}, **kwargs)
    if output_tool and not dry_run:
        query_n8n_service(name, toolname=toolname,
                          output=output_tool, **kwargs)
    return response


if __name__ == "__main__":
    model_names = list(_name2cls.keys())
    parser = argparse.ArgumentParser(
        "Turn server entry points into n8n request forms.")
    subparsers = parser.add_subparsers(
        dest="action", help="Action to perform")
    # Create tool
    parser_create = subparsers.add_parser(
        "create", help="Create an n8n tool")
    parser_update = subparsers.add_parser(
        "update", help="Update an existing tool")
    for x in [parser_create, parser_update]:
        x.add_argument(
            "--name", type=str, nargs="+", action="extend",
            choices=model_names,
            help="Name of the entry point(s) to create tools for."
        )
        x.add_argument(
            "--toolname", type=str,
            help="Name of the tool to create (single name must be provided)",
        )
        x.add_argument(
            "--publish-for-address", type=str,
            help=(
                "Address for the service that should be used in the "
                "published tool"
            ),
        )
        x.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite any existing tool",
        )
        x.add_argument(
            "--dry-run", action="store_true",
            help="Don't actually create the tool.",
        )
        x.add_argument(
            "--output-request", type=str, nargs='?', const=True,
            help="Output the tool creation request to a file",
        )
        x.add_argument(
            "--output-tool", "--output", type=str, nargs='?', const=True,
            help="Output the resulting tool to a file",
        )
        x.add_argument(
            "--output-form", type=str, nargs='?', const=True,
            help="Output the form for a tool to a file",
        )
    parser_create.add_argument(
        "--update", action="store_true",
        help="Update any existing tool",
    )
    # Remove tool
    parser_remove = subparsers.add_parser(
        "remove", help="Remove an n8n tool")
    parser_remove.add_argument(
        "--name", type=str, nargs="+", action="extend",
        choices=model_names,
        help="Name of the entry point(s) to remove tools for."
    )
    parser_remove.add_argument(
        "--toolname", type=str,
        help="Name of the tool to remove",
    )
    parser_remove.add_argument(
        "--output-tool", "--output", type=str, nargs='?', const=True,
        help="Output the removed tool to a file",
    )
    parser_remove.add_argument(
        "--dry-run", action="store_true",
        help="Don't actually remove the tool.",
    )
    # Query
    parser_query = subparsers.add_parser(
        "query", help="Query an n8n tool")
    parser_query.add_argument(
        "--name", type=str, nargs="+", action="extend",
        choices=model_names,
        help="Name of the entry point(s) to query tools for."
    )
    parser_query.add_argument(
        "--toolname", type=str,
        help="Name of the tool to query",
    )
    parser_query.add_argument(
        "--output-tool", "--output", type=str, nargs='?', const=True,
        help="Output the summary for a tool to a file",
    )
    for x in [parser_create, parser_update, parser_remove, parser_query]:
        x.add_argument(
            "--verbose", action="store_true",
            help="Print all REST API responses",
        )
    args = parser.parse_args()
    if not args.name:
        if args.action in ["query", "remove"] and args.toolname:
            args.name = [""]  # Won't be used
        else:
            args.name = model_names
    if len(args.name) > 1:
        assert not args.toolname
        for k in ["output_request", "output_tool", "output_form"]:
            assert not isinstance(getattr(args, k, None), str)
    if args.action in ["create", "update"]:
        if not args.publish_for_address:
            args.publish_for_address = os.environ.get(
                "APSIMX_REMOTE_ADDRESS", None)
    if args.action == "update":
        args.update = "required"
    for name in args.name:
        print(f"{args.action} {name}")
        if args.action in ["create", "update"]:
            publish_n8n_service(
                name,
                service_address=args.publish_for_address,
                toolname=args.toolname,
                overwrite=args.overwrite,
                update=args.update,
                dry_run=args.dry_run,
                output_request=args.output_request,
                output_tool=args.output_tool,
                output_form=args.output_form,
                verbose=args.verbose,
            )
        elif args.action == "remove":
            remove_n8n_service(
                name,
                toolname=args.toolname,
                output=args.output_tool,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        elif args.action == "query":
            response = query_n8n_service(
                name,
                toolname=args.toolname,
                output=args.output_tool,
                allow_multiple=True,
                verbose=args.verbose,
            )
            if not args.output_tool:
                print(json.dumps(response, indent=2))
