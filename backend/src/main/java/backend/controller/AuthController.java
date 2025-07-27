package backend.controller;

import backend.controller.dto.LoginRequest;
import backend.controller.dto.RegisterRequest;
import backend.domain.User;
import backend.service.UserService;
import backend.config.JwtUtil;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final UserService userService;
    private final JwtUtil jwtUtil;

    public AuthController(UserService userService, JwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    /** 일반 회원가입 (USER) */
    @PostMapping("/register")
    public ResponseEntity<User> register(
            @Validated @RequestBody RegisterRequest req
    ) {
        User created = userService.register(req.getUsername(), req.getPassword());
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    /** 일반 로그인 (USER/ADMIN 공용) → JWT 토큰 반환 */
    @PostMapping("/login")
    public ResponseEntity<Map<String,Object>> login(
            @Validated @RequestBody LoginRequest req
    ) {
        User u = userService.login(req.getUsername(), req.getPassword());
        String token = jwtUtil.generateToken(u.getUsername(), u.getRole());
        return ResponseEntity.ok(Map.of(
                "token", token,
                "username", u.getUsername(),
                "role", u.getRole()
        ));
    }

    /** 관리자 전용 로그인 → JWT 토큰 반환 */
    @PostMapping("/login-admin")
    public ResponseEntity<Map<String,Object>> loginAdmin(
            @Validated @RequestBody LoginRequest req
    ) {
        User admin = userService.loginAdmin(req.getUsername(), req.getPassword());
        String token = jwtUtil.generateToken(admin.getUsername(), admin.getRole());
        return ResponseEntity.ok(Map.of(
                "token", token,
                "username", admin.getUsername(),
                "role", admin.getRole()
        ));
    }

    /** 전체 사용자 조회 (관리자 토큰 필요) */
    @GetMapping("/users")
    public ResponseEntity<List<User>> listUsers() {
        return ResponseEntity.ok(userService.findAll());
    }
}
