from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional, Literal, Tuple, Any


RenderMode = Optional[Literal["human", "rgb_array"]]

class PixelCopterRenderWrapper(gym.Wrapper):
    """
    Add modern render/render_mode to PLE-based gym_pygame envs from qlan3/gym-games,
    e.g. 'PixelCopter-PLE-v0'.

    - render_mode="rgb_array": returns an HxWx3 uint8 frame from the PLE screen.
    - render_mode="human": opens a pygame window and blits frames to it.
    - render_mode=None: no rendering (default Gymnasium behavior).

    Notes:
      * Underlying PLE object is discovered dynamically; it must expose getScreenRGB().
      * This wrapper does not change observations/rewards/terminations; it only augments rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env: gym.Env, render_mode: RenderMode = None):
        super().__init__(env)
        self._render_mode: RenderMode = render_mode
        # Try to sniff a render fps from the base env if present
        base_fps = getattr(getattr(env, "metadata", {}), "get", lambda *_: None)("render_fps")
        if isinstance(base_fps, (int, float)) and base_fps > 0:
            self.metadata["render_fps"] = int(base_fps)

        # pygame resources (created lazily if needed)
        self._pygame_inited = False
        self._display_surf = None

    # ---------- Gymnasium API passthrough ----------
    def reset(self, **kwargs) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        if self._render_mode == "human":
            self._render_human_init_if_needed()
            self.render()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._render_mode is not None:
            self.render()
        return obs, reward, terminated, truncated, info

    # ---------- Render helpers ----------
    def render(self):
        if self._render_mode is None:
            return None

        rgb = self._get_rgb_frame()  # H x W x 3, uint8

        if self._render_mode == "rgb_array":
            return rgb

        if self._render_mode == "human":
            self._render_human_init_if_needed()
            import pygame

            # Convert numpy frame to a surface and blit
            frame = np.transpose(rgb, (1, 0, 2))  # surface expects W x H x 3 when using surfarray
            surf = pygame.surfarray.make_surface(frame)
            # Resize surface to window if sizes differ
            if self._display_surf.get_size() != (surf.get_width(), surf.get_height()):
                surf = pygame.transform.smoothscale(surf, self._display_surf.get_size())
            self._display_surf.blit(surf, (0, 0))
            pygame.display.flip()
            # Pump events so the window remains responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
            return None

        raise NotImplementedError(f"Unsupported render_mode: {self._render_mode}")

    def close(self):
        if self._pygame_inited:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._pygame_inited = False
            self._display_surf = None
        return super().close()

    # ---------- Internals ----------
    def _get_ple(self):
        """
        Try to locate the underlying PLE instance which should expose getScreenRGB().
        Common attributes used by PLE wrappers: p, ple, _ple, game (sometimes), etc.
        """
        cand_attrs = ["p", "ple", "_ple", "_p", "viewer", "engine", "game"]
        unwrapped = self.env.unwrapped
        for name in cand_attrs:
            obj = getattr(unwrapped, name, None)
            if obj is not None and hasattr(obj, "getScreenRGB"):
                return obj
            # Some envs store the PLE under .p but the game object under .game; both may implement getScreenRGB
            if obj is not None and hasattr(obj, "game") and hasattr(getattr(obj, "game"), "getScreenRGB"):
                return getattr(obj, "game")
        # As a last resort, see if the env itself implements getScreenRGB
        if hasattr(unwrapped, "getScreenRGB"):
            return unwrapped
        raise RuntimeError(
            "Could not find a PLE-like object with getScreenRGB() on the underlying env. "
            "Ensure you are using a PLE-based env from qlan3/gym-games and that PLE is installed."
        )

    def _get_rgb_frame(self) -> np.ndarray:
        ple = self._get_ple()
        rgb = ple.getScreenRGB()  # expected shape (W, H, 3) or (H, W, 3) depending on backend
        rgb = np.asarray(rgb)
        # Normalize to H x W x 3 uint8
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        # Heuristic: if first dim is wider than second dim by typical landscape aspect, assume (W, H, 3)
        if rgb.ndim == 3 and rgb.shape[0] > rgb.shape[1]:
            # Looks like (W, H, 3) -> transpose to (H, W, 3)
            rgb = np.transpose(rgb, (1, 0, 2))
        return rgb

    def _render_human_init_if_needed(self):
        if self._pygame_inited:
            return
        import pygame

        frame = self._get_rgb_frame()
        h, w = frame.shape[:2]
        pygame.init()
        # Create a window roughly the game’s native size (can be changed here)
        self._display_surf = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Gym PLE Viewer")
        self._pygame_inited = True
    
    @property
    def render_mode(self) -> RenderMode:
        return self._render_mode
