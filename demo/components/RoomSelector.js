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
            <form onSubmit={this.handleSubmit}>
                <label>
                    Username:
                    <input type="text" value={this.state.username} onChange={this.handleUsername} />
                </label>
                <label>
                    Chatroom Name:
                    <input type="text" value={this.state.room} onChange={this.handleRoom} />
                </label>
                <input type="submit" value="Submit" />
            </form>

        )
    }
}