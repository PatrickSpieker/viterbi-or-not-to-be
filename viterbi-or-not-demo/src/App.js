import React, {Component} from 'react';

import RoomSelector from './RoomSelector';
import MainInterface from './MainInterface';

import firebase from 'firebase/app';
import 'firebase/firestore';

import { getConfig } from './environment.js';

export default class App extends Component {
    constructor(props) {
        super(props);

        let config = getConfig();
        firebase.initializeApp(config);

        const db = firebase.firestore();
        db.settings({
            timestampsInSnapshots: true
        });

        this.state = {
            username: null,
            room: null,
            db: db
        }

        this.selectRoom = this.selectRoom.bind(this);
    }

    selectRoom(newUsername, newRoom) {
        this.setState({
            username: newUsername,
            room: newRoom
        });
    }

    render() {
        if (this.state.room === null) {
            return <RoomSelector selectRoom={this.selectRoom} />
        } else {
            return <MainInterface db={this.state.db} room={this.state.room} username={this.state.username} />
        }
    }
}
