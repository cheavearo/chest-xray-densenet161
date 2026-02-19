def format_result_html(results):
    best_label, best_prob = results[0]

    rows = ""
    for label, prob in results:
        percent = prob * 100

        rows += f"""
        <div class="row">
            <div class="row-header">
                <span class="label">{label.replace('_',' ').title()}</span>
                <span class="percent">{percent:.2f}%</span>
            </div>
            <div class="progress">
                <div class="progress-fill" style="width:{percent:.2f}%"></div>
            </div>
        </div>
        """

    html = f"""
    <div class="report">
        <div class="diagnosis">
            <div class="diag-title">Prediction</div>
            <div class="diag-main">{best_label.replace('_',' ').title()}</div>
            <div class="diag-sub">Confidence {best_prob*100:.2f}%</div>
        </div>

        {rows}
    </div>
    """
    return html
