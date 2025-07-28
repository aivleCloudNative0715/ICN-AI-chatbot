package backend.controller;

import backend.controller.dto.LoginRequest;
import backend.controller.dto.RegisterRequest;
import backend.domain.User;
import backend.service.UserService;
import backend.service.TokenBlacklist;
import backend.config.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
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
    private final TokenBlacklist blacklist;

    public AuthController(
            UserService userService,
            JwtUtil jwtUtil,
            TokenBlacklist blacklist
    ) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
        this.blacklist = blacklist;
    }

    /** 일반 회원가입 (USER) */
    @PostMapping("/register")
    public ResponseEntity<User> register(@Validated @RequestBody RegisterRequest req) {
        User created = userService.register(req.getUsername(), req.getPassword());
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    /** 일반 로그인 (USER/ADMIN 공용) → JWT 토큰 반환 */
    @PostMapping("/login")
    public ResponseEntity<Map<String,Object>> login(@Validated @RequestBody LoginRequest req) {
        User u = userService.login(req.getUsername(), req.getPassword());
        String token = jwtUtil.generateToken(u.getUsername(), u.getRole());
        return ResponseEntity.ok(Map.of(
                "token",    token,
                "username", u.getUsername(),
                "role",     u.getRole()
        ));
    }

    /** 관리자 전용 로그인 → JWT 토큰 반환 */
    @PostMapping("/login-admin")
    public ResponseEntity<Map<String,Object>> loginAdmin(@Validated @RequestBody LoginRequest req) {
        User admin = userService.loginAdmin(req.getUsername(), req.getPassword());
        String token = jwtUtil.generateToken(admin.getUsername(), admin.getRole());
        return ResponseEntity.ok(Map.of(
                "token",    token,
                "username", admin.getUsername(),
                "role",     admin.getRole()
        ));
    }

    /** 로그아웃: 토큰을 블랙리스트에 추가하고 메시지 반환 */
    @PostMapping("/logout")
    public ResponseEntity<Map<String,String>> logout(HttpServletRequest req) {
        String header = req.getHeader("Authorization");
        if (header != null && header.startsWith("Bearer ")) {
            blacklist.add(header.substring(7));
        }
        return ResponseEntity.ok(Map.of("message", "로그아웃 되었습니다."));
    }

    /** 전체 사용자 조회 (ADMIN 토큰 필요) */
    @GetMapping("/users")
    public ResponseEntity<List<User>> listUsers() {
        return ResponseEntity.ok(userService.findAll());
    }
}
