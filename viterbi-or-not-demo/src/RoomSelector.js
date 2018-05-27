import React, {Component} from 'react';

export default class RoomSelector extends Component {
    constructor(props) {
        super(props);

        this.state = {
            username: '',
            room: ''
        }

        this.handleUsername = this.handleUsername.bind(this);
        this.handleRoom = this.handleRoom.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleUsername(event) {
        this.setState({username: event.target.value});
    }

    handleRoom(event) {
        this.setState({room: event.target.value});
    }

    handleSubmit(event) {
        event.preventDefault();
        this.props.selectRoom(this.state.username, this.state.room);
    }

    render() {
        return (
            <div id="room-selector-container">
                <div id="room-selector">
                    <form onSubmit={this.handleSubmit}>
                        <div id="room-selector-username">
                            <label>
                                <h2 className="form-label">Username</h2>
                                <div className="form-line">
                                    <input className="form-input" type="text" value={this.state.username} onChange={this.handleUsername} />
                                </div>
                            </label>
                        </div>
                        <div id="room-selector-room">
                            <label>
                                <h2 className="form-label">Chatroom Name</h2>
                                <div className="form-line">
                                    <input className="form-input" type="text" value={this.state.room} onChange={this.handleRoom} />
                                    <button className="submit-button" type="submit"><i className="material-icons">arrow_forward</i></button>
                                </div>
                            </label>
                            <div id="room-selector-existing">
                                <h2 className="form-label">Or Select An Existing Chat</h2>
                                <div id="room-selector-options">
                                    <input type="button" value="Dog Walking" />
                                    <input type="button" value="Throwing People Into Buckets" />
                                    <input type="button" value="We Broke the Fridge!" />
                                    <input type="button" value="Look at that Inequality" />
                                </div>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        )
    }
}