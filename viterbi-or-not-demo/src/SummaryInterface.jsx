import React, {Component} from 'react';
import Slider, { createSliderWithTooltip } from 'rc-slider';
import 'rc-slider/assets/index.css';

const SliderWithTooltip = createSliderWithTooltip(Slider);

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
            if (this.props.features[feature].some(x => x !== 0)) {
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
                    );
                }
            }
        }

        let messageLog = [];
        for (let i = 0; i < this.props.summary.length; i++) {
            // let background = {backgroundColor: 'rgba(5, 168, 170, ' + this.props.predictions[i] + ')'};
            let message = this.props.summary[i];

            let originalIndex = this.props.summaryMap[i];
            console.log('original index ' + originalIndex);
            let newTopic = this.props.features['topic_position'][originalIndex] === 0 ? 'topic-start' : '';
            messageLog.push(
                <li key={message.key} className={newTopic} /*style={background}*/>{message}</li>
            );
        }

        return (
            <div id="summary-interface">
                <h3>FEATURES TO VISUALIZE</h3>
                <div id="feature-options">{summaryOptions}</div>
                <h3>COMPRESSION RATIO</h3>
                <div id="slider">
                    <SliderWithTooltip
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        value={this.props.threshold}
                        onChange={this.props.adjustThreshold}
                    />
                </div>
                <h3>SUMMARY</h3>
                <div id="summary-listing-container">
                    <ul id="summary-listing">{messageLog}</ul>
                </div>
            </div>
        );
    }
}