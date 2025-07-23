package backend.controller;

import backend.controller.dto.RegisterRequest;
import backend.domain.User;
import backend.service.UserService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;  // 추가

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final UserService userService;

    public AuthController(UserService userService) {
        this.userService = userService;
    }

    /** 회원가입 */
    @PostMapping("/register")
    public ResponseEntity<User> register(
            @Validated @RequestBody RegisterRequest req
    ) {
        User created = userService.register(req.getUsername(), req.getPassword());
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    /** 전체 사용자 조회 */
    @GetMapping("/users")
    public ResponseEntity<List<User>> listUsers() {
        List<User> users = userService.findAll();
        return ResponseEntity.ok(users);
    }
}