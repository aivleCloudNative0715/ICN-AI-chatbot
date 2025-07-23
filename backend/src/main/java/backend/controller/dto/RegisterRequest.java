package backend.controller.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data                             // ← Lombok이 getter/setter, toString 등을 자동 생성
public class RegisterRequest {
    @NotBlank
    private String username;

    @NotBlank
    private String password;
}
