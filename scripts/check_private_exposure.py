"""Read-only checks for public routes/ingresses to this keyless development service.

The workflow removes only the old, known Route named marker-databricks.
Unexpected exposure is reported for an operator; unrelated routes are not deleted.
NetworkPolicy enforcement and additive allow policies still require admin review.
"""
import json
import subprocess

SERVICE = 'marker-databricks'


def exposures(routes, ingresses):
    found = []
    for route in routes.get('items', []):
        spec = route.get('spec', {})
        backends = [spec.get('to', {}), *spec.get('alternateBackends', [])]
        if any(b.get('name') == SERVICE and b.get('kind', 'Service') == 'Service' for b in backends):
            found.append('Route/' + route['metadata']['name'])
    for ingress in ingresses.get('items', []):
        spec = ingress.get('spec', {})
        backends = [spec.get('defaultBackend', {}), spec.get('backend', {})]
        for rule in spec.get('rules', []):
            for path in rule.get('http', {}).get('paths', []):
                backends.append(path.get('backend', {}))
        if any(b.get('service', {}).get('name', b.get('serviceName')) == SERVICE for b in backends):
            found.append('Ingress/' + ingress['metadata']['name'])
    return sorted(found)


def read_list(resource):
    result = subprocess.run(['oc', 'get', resource, '-o', 'json'], text=True,
                            capture_output=True, check=False)
    if result.returncode:
        # Do not print command output that could include sensitive cluster context.
        raise RuntimeError(f'Cannot list {resource}; private-access verification requires permission')
    return json.loads(result.stdout)


def main():
    found = exposures(read_list('routes'), read_list('ingresses'))
    if found:
        raise SystemExit('Refusing keyless deployment with public exposure: ' + ', '.join(found) +
                         '. Remove or disconnect these routes from this service, then rerun.')
    print('No Route/Ingress directly references the keyless gateway service.')


if __name__ == '__main__':
    main()
