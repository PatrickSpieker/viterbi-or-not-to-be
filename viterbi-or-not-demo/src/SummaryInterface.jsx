import React, {Component} from 'react';

export default class SummaryInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
        }
    }

    render() {
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
                <ul id="summary-listing">{messageLog}</ul>
            </div>
        );
    }
}