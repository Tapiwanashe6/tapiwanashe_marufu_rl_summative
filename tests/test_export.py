from visualization.export_events import export


def test_export_events(tmp_path):
    path = tmp_path / 'events.json'
    data = export(seed=42, output=path)
    assert path.exists()
    assert len(data['sites']) == 11
    assert len(data['events']) > 0
    assert data['events'][0]['before'] == data['initial']
    for e in data['events']:
        assert e['end_time'] >= e['start_time']
        if e['flight']:
            assert e['destination'] == e['action'] - 1
            assert e['event'] in ('delivered', 'late_delivery')
    assert data['summary']['steps'] == len(data['events'])
