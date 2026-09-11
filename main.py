from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video


def main():
    """read video"""
    video_frames = read_video("videos/video0.mp4")

    """get object track"""
    tracker = Tracker("models/best_m.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    """interpolate ball positions"""
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    """Assign Player Teams"""
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    """Draw output"""
    # Draw Output Tracks
    output_video_frames = tracker.draw_annotation(video_frames, tracks)

    """save video"""
    save_video(output_video_frames, "videos/video0_output.avi")
    print("saved")


if __name__ == "__main__":
    main()
