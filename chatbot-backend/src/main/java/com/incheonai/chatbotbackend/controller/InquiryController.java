package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.InquiryRequestDto;
import com.incheonai.chatbotbackend.service.InquiryService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/board")
public class InquiryController {

    private final InquiryService inquiryService;

    /** 전체 문의/건의 목록 조회 (게시판 기능) */
    @GetMapping
    public ResponseEntity<Page<InquiryDto>> getAllInquiries(
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search,
            Pageable pageable
    ) {
        Page<InquiryDto> inquiries = inquiryService.getAllInquiries(category, search, pageable);
        return ResponseEntity.ok(inquiries);
    }

    /** 내 문의 목록 조회 */
    @GetMapping("/my")
    public ResponseEntity<Page<InquiryDto>> getMyInquiries(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search,
            Pageable pageable
    ) {
        String userId = userDetails.getUsername();
        Page<InquiryDto> inquiries = inquiryService.getMyInquiries(userId, category, search, pageable);
        return ResponseEntity.ok(inquiries);
    }

    /** 문의 상세 조회*/
    @GetMapping("/{inquiryId}")
    public ResponseEntity<InquiryDetailDto> getInquiryDetail(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable Integer inquiryId
    ) {
        // 2. userId를 넘기지 않고, 새로 만든 서비스 메서드를 호출합니다.
        InquiryDetailDto detail = inquiryService.getInquiryDetailById(inquiryId);
        return ResponseEntity.ok(detail);
    }

    /** 문의 생성 */
    @PostMapping
    public ResponseEntity<InquiryDto> createInquiry(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestBody InquiryRequestDto requestDto
    ) {
        String userId = userDetails.getUsername();
        InquiryDto response = inquiryService.createInquiry(userId, requestDto);
        return ResponseEntity.ok(response);
    }

    /** 내 문의 수정 */
    @PutMapping("/{inquiryId}")
    public ResponseEntity<Map<String, String>> updateMyInquiry(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable Integer inquiryId,
            @RequestBody InquiryRequestDto requestDto
    ) {
        String userId = userDetails.getUsername();
        inquiryService.updateMyInquiry(userId, inquiryId, requestDto);
        return ResponseEntity.ok(Map.of("message", "문의/건의가 수정되었습니다."));
    }

    /** 내 문의 삭제 */
    @DeleteMapping("/{inquiryId}")
    public ResponseEntity<Map<String, String>> deleteMyInquiry(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable Integer inquiryId
    ) {
        String userId = userDetails.getUsername();
        inquiryService.deleteMyInquiry(userId, inquiryId);
        return ResponseEntity.ok(Map.of("message", "문의/건의가 삭제되었습니다."));
    }
}
