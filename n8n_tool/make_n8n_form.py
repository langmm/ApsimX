import os
import json
import pprint
import argparse
import requests
from pydantic import BaseModel
from config import ModelInput, InteractiveModelInput


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
        "fieldLabel": field_name.replace("_", " ").title(),
        "fieldName": field_name,
        "fieldType": n8n_type,
    }
    if n8n_type == "date":
        field_def["formatDate"] = "YYYY-MM-DD"
    if details.get("description", None) is not None:
        field_def["fieldLabel"] += " " + details["description"]
    if details.get("default", None) is not None:
        field_default = details["default"]
        if isinstance(field_default, list):
            if n8n_type not in ["checkbox"]:
                field_default = ",".join(field_default)
        elif isinstance(field_default, dict):
            field_default = json.dumps(field_default)
        if n8n_type not in ["dropdown", "checkbox", "radio", "date", "file"]:
            field_def["placeholder"] = field_default
        if n8n_type not in ["password", "html", "hiddenField", "file"]:
            field_def["defaultValue"] = field_default
    if pydantic_enum:
        field_def["fieldOptions"] = {
            "values": [{"option": k} for k in pydantic_enum]
        }
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


def name_to_n8n_form(name: str):
    model = _name2cls[name]
    fname = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f"n8n-form-apsimx-{name}.json"
    )
    with open(fname, "w") as fd:
        json.dump(pydantic_to_n8n_form(name, model), fd, indent=2)
    return fname


def n8n_api_request(path, action, headers=None, **kwargs):
    if headers is None:
        headers = {}
    credentials = headers.get(
        'X_N8N_API_KEY', os.environ.get('X_N8N_API_KEY', None))
    if credentials is None:
        raise RuntimeError("No credentials provided and "
                           "\"X_N8N_API_KEY\" environment "
                           "variable not set")
    headers['X-N8N-API-KEY'] = credentials
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
    print(action.upper(), f'{_n8n_address}/{path}')
    pprint.pprint(out)
    return out


def remove_n8n_service(name: str, toolname: str | None = None,
                       **kwargs):
    if toolname is None:
        toolname = f'ApsimX {name} Tool'
    response = n8n_api_request(
        'workflows', 'get', params={'name': toolname}, **kwargs
    )
    if len(response['data']) == 0:
        raise RuntimeError(
            f"No tool found matching name \"{toolname}\"")
    elif len(response['data']) > 1:
        raise RuntimeError(
            f"More than one tool matching name \"{toolname}\":\n"
            f"{pprint.pformat(response['data'])}")
    idstr = response['data'][0]["id"]
    return n8n_api_request(
        f'workflows/{idstr}', 'delete', **kwargs
    )


def publish_n8n_service(service_address: str, name: str,
                        fname_form: str,
                        toolname: str | None = None,
                        outputfile: str | None = None,
                        **kwargs):
    if toolname is None:
        toolname = f'ApsimX {name} Tool'
    # Check if the workflow exists
    response = n8n_api_request(
        'workflows', 'get', params={'name': toolname}, **kwargs
    )
    if len(response['data']) > 0:
        raise RuntimeError(
            f"Tool(s) already exists with name \"{toolname}\":\n"
            f"{pprint.pformat(response['data'])}")
    with open(fname_form, 'r') as fd:
        form_node = json.load(fd)
    nodes = [
        form_node,
        {
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
            "position": [820, 320],
        }
    ]
    request = {
        "name": toolname,
        "nodes": nodes,
        "connections": {
            form_node["name"]: {
                "main": [[{
                    "node": "HTTP Request",
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
    }
    if outputfile is not None:
        with open(outputfile, 'w') as fd:
            json.dump(request, fd)
        return
    return n8n_api_request(
        'workflows', 'post', json=request,
        headers={'accept': 'application/json'}, **kwargs)


if __name__ == "__main__":
    model_names = list(_name2cls.keys())
    parser = argparse.ArgumentParser(
        "Turn server entry points into n8n request forms.")
    parser.add_argument(
        "--name", type=str, nargs="+", action="extend",
        choices=model_names,
        help="Name of the entry point to create a form for."
    )
    parser.add_argument(
        "--publish-for-address", type=str,
        help=(
            "Address for the service that should be used in the "
            "published tool"
        ),
    )
    args = parser.parse_args()
    if not args.name:
        args.name = model_names
    for name in args.name:
        fname = name_to_n8n_form(name)
        if args.publish_for_address:
            publish_n8n_service(args.publish_for_address, name, fname)
