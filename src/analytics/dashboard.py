"""
@Author: Raúl Javierre
@Date: updated 26/05/2021

This module provides an interactive dashboard (http://localhost:8050) that:
- Shows information about the experimentation (GiB used, time spent, carbon footprint generated...)
- Compares the detectors under different metrics (graphics and tables)
- Compares the scenarios (graphics and tables)
"""

import os
import sys
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from src.analytics import generate_file, CSV_FILE
from src.analytics.MetadataInformation import MetadataInformation
from src.analytics.components.ComponentFactory import ComponentFactory


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

components = {
    'AccComponent': ComponentFactory.create_component('AccComponent'),
    'MccComponent': ComponentFactory.create_component('MccComponent'),
    'BaccComponent': ComponentFactory.create_component('BaccComponent'),
    'TprComponent': ComponentFactory.create_component('TprComponent'),
    'TnrComponent': ComponentFactory.create_component('TnrComponent'),
    'TbComponent': ComponentFactory.create_component('TbComponent'),
    'TpComponent': ComponentFactory.create_component('TpComponent'),
    'DetectorsTableComponent': ComponentFactory.create_component('DetectorsTableComponent'),
    'NormalVsAveragesComponent': ComponentFactory.create_component('NormalVsAveragesComponent'),
    'NormalVsFDIsComponent': ComponentFactory.create_component('NormalVsFDIsComponent'),
    'NormalVsSwapComponent': ComponentFactory.create_component('NormalVsSwapComponent'),
    'NormalVsRSAsComponent': ComponentFactory.create_component('NormalVsRSAsComponent'),
    'NormalVsRatingVsPercentileComponent': ComponentFactory.create_component('NormalVsRatingVsPercentileComponent'),
    'ScenariosTableComponent': ComponentFactory.create_component('ScenariosTableComponent'),
}

selected_dataset = 'Electricity (ISSDA-CER)'
selected_scenario = 'All'


@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('dropdown-dataset', 'value'), dash.dependencies.Input('dropdown-scenario', 'value')])
def update_page_content(dataset, scenario):

    for component in components.values():
        component.update(dataset, scenario)

    return layout_function(dataset, scenario)


def layout_function(dataset=selected_dataset, scenario=selected_scenario):
    return html.Div([
            html.Div([

                html.Div([
                    html.Div([
                        html.H5('Choose a dataset'),
                        dcc.Dropdown(
                            id='dropdown-dataset',
                            options=[
                                {'label': 'Electricity (ISSDA-CER)', 'value': 'Electricity (ISSDA-CER)'},
                                {'label': 'Gas (ISSDA-CER)', 'value': 'Gas (ISSDA-CER)'},
                                {'label': 'Solar (Ausgrid)', 'value': 'Solar (Ausgrid)'},
                            ],
                            value=dataset
                        ),
                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

                html.Div([
                    html.Div([
                        html.H5('Choose a scenario'),
                        dcc.Dropdown(
                            id='dropdown-scenario',
                            options=[
                                {'label': 'All', 'value': 'All'},
                                {'label': 'Normal', 'value': 'Normal'},
                                {'label': 'RSA_0.25_1.1', 'value': 'RSA_0.25_1.1'},
                                {'label': 'RSA_0.5_1.5', 'value': 'RSA_0.5_1.5'},
                                {'label': 'RSA_0.5_3', 'value': 'RSA_0.5_3'},
                                {'label': 'Avg', 'value': 'Avg'},
                                {'label': 'Min-Avg', 'value': 'Min-Avg'},
                                {'label': 'Swap', 'value': 'Swap'},
                                {'label': 'FDI0', 'value': 'FDI0'},
                                {'label': 'FDI5', 'value': 'FDI5'},
                                {'label': 'FDI10', 'value': 'FDI10'},
                                {'label': 'FDI20', 'value': 'FDI20'},
                                {'label': 'FDI30', 'value': 'FDI30'},
                            ],
                            value=scenario
                        ),
                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

                html.Div([
                    html.Div([
                        html.H4('Summary', style={'marginBottom': '3%'}),
                        html.P('💾 Selected dataset: ' + dataset),
                        html.P('📈 Selected scenario: ' + scenario),
                        html.P('🏡 Number of processed meterIDs: ' + str(MetadataInformation.get_number_of_meterIDs(dataset))),
                        html.P('🦾 Number of training weeks: ' + str(MetadataInformation.get_number_of_training_weeks(dataset))),
                        html.P('️⚙ Number of testing weeks: ' + str(MetadataInformation.get_number_of_testing_weeks(dataset))),
                        html.P('🤖 Number of detectors: ' + str(MetadataInformation.get_number_of_detectors(dataset))),
                        html.P('😀️️ Number of normal scenarios: ' + str(MetadataInformation.get_number_of_normal_scenarios(dataset))),
                        html.P('⚔️️ Number of attack scenarios: ' + str(MetadataInformation.get_number_of_attack_scenarios(dataset))),
                        html.P('⚖️️️ Training dataset: ' + str(MetadataInformation.get_gb_of_training_dataset(dataset)) + ' GiB'),
                        html.P('⚖️️️ Testing dataset: ' + str( MetadataInformation.get_gb_of_testing_dataset(dataset)) + ' GiB'),
                        html.P('⌚️️ Experimentation time: ' + str(MetadataInformation.get_experimentation_time(dataset)) + ' days'),
                        html.P('🏭️️ Carbon footprint generated: ' + str(MetadataInformation.get_carbon_footprint(dataset)) + ' kg. of CO2'),
                        html.P('️🌳 Amount of trees needed: ' + str(MetadataInformation.get_number_of_trees(dataset)) + ' trees')
                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

                html.Div([
                    html.Div([
                        html.H1('Analysis of detectors', style={'margin': '3%'}),

                        html.Div([
                            html.H5(
                                "Using the metric 'Accuracy' (acc)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['AccComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Matthews Correlation Coefficient' (mcc)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['MccComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Balanced accuracy' (bacc)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['BaccComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Recall = Sensitivity' (tpr)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['TprComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Specificity' (tnr)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['TnrComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Time to build the model' (tb)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['TbComponent'].component,

                        html.Div([
                            html.H5(
                                "Using the metric 'Time to predict with the model' (tp)",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%'}),
                        components['TpComponent'].component,

                        html.Div([
                            html.H5(
                                "Comparative table",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%', 'marginBottom': '3%'}),
                        components['DetectorsTableComponent'].component,

                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

                html.Div([
                    html.Div([
                        html.H1('Comparison of the curves of the different scenarios', style={'margin': '3%'}),
                        html.Div([
                            html.H5(
                                "How different are the scenarios?",
                                style={'textAlign': 'center', 'marginTop': '8%', 'fontWeight': 'bold'}),
                            html.H5(
                                "Here we can see the evolution of the usage of one meterID in one week under different scenarios",
                                style={'textAlign': 'center'}),
                        ], style={'marginLeft': '5%', 'marginRight': '5%', 'marginBottom': '3%'}),
                        components['NormalVsAveragesComponent'].component,
                        components['NormalVsFDIsComponent'].component,
                        components['NormalVsSwapComponent'].component,
                        components['NormalVsRSAsComponent'].component,
                        components['NormalVsRatingVsPercentileComponent'].component,
                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

                html.Div([
                    html.Div([
                        html.H1('Economic benefit for each scenario 💰', style={'margin': '3%'}),
                        html.H5(
                            "How much money does an average attacker save in 60 weeks?",
                            style={'textAlign': 'center', 'marginLeft': '5%', 'marginRight': '5%',
                                   'marginBottom': '3%'}),
                        components['ScenariosTableComponent'].component,
                    ], className="p-4 card-body p-0"),
                ], className="card shadow-lg border-1 my-4 card"),

            ], className="container", style={'fontFamily': 'Nunito'})
        ], id="page-content")


if __name__ == '__main__':
    """
    args:
    sys.argv[1]:regenerate (if you want to update the dashboard with new data)
    """

    if not os.path.exists(CSV_FILE) or len(sys.argv) == 2 and sys.argv[1] == "regenerate":
        generate_file()

    app.layout = layout_function

    app.run_server(debug=True)
