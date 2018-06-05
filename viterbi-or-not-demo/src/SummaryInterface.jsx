import React, {Component} from 'react';

export default class SummaryInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
        }

        this.handleFeatureChange = this.handleFeatureChange.bind(this);
    }

    handleFeatureChange(event) {
        this.props.toggleFeature(event.target.getAttribute('value'));
    }

    render() {
        let summaryOptions = [];
        let selectedClasses = ['first', 'second', 'third', 'fourth'];
        for (let feature of Object.keys(this.props.features)) {
            if (this.props.selectedFeatures.includes(feature)) {
                summaryOptions.push(
                    <button id={'feature-selector-' + feature} className={'feature-selector ' + selectedClasses[this.props.selectedFeatures.indexOf(feature)]} key={feature} value={feature} type="button" onClick={this.handleFeatureChange}>
                        <i className="material-icons">check_box</i>
                        {feature}
                    </button>
                );
            } else {
                summaryOptions.push(
                    <button id={'feature-selector-' + feature} className={'feature-selector'} key={feature} value={feature} type="button" onClick={this.handleFeatureChange}>
                        <i className="material-icons">check_box_outline_blank</i>
                        {feature}
                    </button>
                )
            }
        }

        let messageLog = [];
        for (let i = 0; i < this.props.summary.length; i++) {
            // let background = {backgroundColor: 'rgba(5, 168, 170, ' + this.props.predictions[i] + ')'};
            let message = this.props.summary[i];

            messageLog.push(
                <li key={message.key} /*style={background}*/>{message}</li>
            )
        }

        return (
            <div id="summary-interface">
                <div id="feature-options">{summaryOptions}</div>
                <div id="summary-listing-container">
                    <ul id="summary-listing">{messageLog}</ul>
                </div>
            </div>
        );
    }
}