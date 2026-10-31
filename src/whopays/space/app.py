"""Run with python -m whopays.space.app after installing whopays[space]."""
import os

from whopays.report.analysis import load_parquet


def create_app(root):
    import gradio as gr

    rows = load_parquet(root)
    cols = [c for c in ['run_id', 'model', 'task', 'lang', 'weight', 'kv', 'spec',
                       'calib', 'seed', 'correct', 'input_tokens', 'output_tokens',
                       'accepted_tokens', 'steps', 'wall_time_s', 'gpu'] if c in rows]

    def select(model, language, run_id):
        frame = rows
        for column, value in [('model', model), ('lang', language), ('run_id', run_id)]:
            if value != 'All':
                frame = frame[frame[column] == value]
        return frame[cols].head(1000)

    with gr.Blocks(title='WhoPays') as app:
        gr.Markdown('# WhoPays\nPer-item results from logged GPU runs. '
                    'Fake-quant memory savings are theoretical. Display limited to 1,000 rows.')
        model = gr.Dropdown(['All', *sorted(rows.model.unique())], value='All', label='Model')
        language = gr.Dropdown(['All', *sorted(rows.lang.unique())], value='All', label='Language')
        config = gr.Dropdown(['All', *sorted(rows.run_id.unique())], value='All', label='Run')
        table = gr.Dataframe(value=rows[cols].head(1000), interactive=False)
        for control in (model, language, config):
            control.change(select, [model, language, config], table)
    return app


if __name__ == '__main__':
    create_app(os.environ.get('WHOPAYS_RELEASE', 'results')).launch()
