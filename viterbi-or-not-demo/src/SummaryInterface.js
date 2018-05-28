import React, {Component} from 'react';

export default class SummaryInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
        }
    }

    render() {
        let messageLog = this.props.summary.map((message) =>
            (<li key={message.key}>{message}</li>)
        );

        return (
            <div id="summary-interface">
                <ul>{messageLog}</ul>
            </div>
        );
    }
}