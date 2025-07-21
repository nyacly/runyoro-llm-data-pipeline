import os
import logging

try:
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
except ImportError:  # pragma: no cover - optional dependency
    ExecuteTask = None
    Task = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def align_audio_text(audio_path, text_path, output_dir, language="eng"):
    """Align audio with text using aeneas if available.

    Parameters
    ----------
    audio_path : str
        Path to the standardized audio file.
    text_path : str
        Path to the processed transcript text file.
    output_dir : str
        Directory where the alignment JSON will be stored.
    language : str
        Language code for aeneas (default "eng").

    Returns
    -------
    str or None
        Path to the generated alignment JSON, or ``None`` if alignment failed or
        ``aeneas`` is not installed.
    """
    if ExecuteTask is None or Task is None:
        logging.warning("aeneas not installed; skipping forced alignment")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(audio_path))[0]}_alignment.json",
    )

    config_string = (
        f"task_language={language}|is_text_type=plain|os_task_file_format=json"
    )
    try:
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = os.path.abspath(audio_path)
        task.text_file_path_absolute = os.path.abspath(text_path)
        task.sync_map_file_path_absolute = os.path.abspath(output_json)

        ExecuteTask(task).execute()
        task.output_sync_map_file()
        logging.info(f"Alignment saved to {output_json}")
        return output_json
    except Exception as e:  # pragma: no cover - runtime safeguard
        logging.error(f"Forced alignment failed for {audio_path} and {text_path}: {e}")
        return None
