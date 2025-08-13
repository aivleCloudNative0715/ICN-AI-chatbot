package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.InquiryRequestDto;
import com.incheonai.chatbotbackend.service.InquiryService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/board")
public class InquiryController {

    private final InquiryService inquiryService;

    /** 전체 문의/건의 목록 조회 (게시판 기능) */
    @GetMapping
    public ResponseEntity<Page<InquiryDto>> getAllInquiries(Pageable pageable) {
        // InquiryService에 모든 문의를 조회하는 메서드(findAll) 추가 필요
        Page<InquiryDto> inquiries = inquiryService.getAllInquiries(pageable);
        return ResponseEntity.ok(inquiries);
    }

    /** 문의 생성 */
    @PostMapping
    public ResponseEntity<InquiryDto> createInquiry(
            @RequestParam String userId,
            @RequestBody InquiryRequestDto requestDto
    ) {
        InquiryDto response = inquiryService.createInquiry(userId, requestDto);
        return ResponseEntity.ok(response);
    }

    /** 내 문의 목록 조회 */
    @GetMapping
    public ResponseEntity<Page<InquiryDto>> getMyInquiries(
            @RequestParam String userId,
            @RequestParam(required = false) String status,
            Pageable pageable
    ) {
        Page<InquiryDto> inquiries = inquiryService.getMyInquiries(userId, status, pageable);
        return ResponseEntity.ok(inquiries);
    }

    /** 내 문의 상세 조회 */
    @GetMapping("/{inquiryId}")
    public ResponseEntity<InquiryDetailDto> getMyInquiryDetail(
            @RequestParam String userId,
            @PathVariable Integer inquiryId
    ) {
        InquiryDetailDto detail = inquiryService.getMyInquiryDetail(userId, inquiryId);
        return ResponseEntity.ok(detail);
    }

    /** 내 문의 수정 */
    @PutMapping("/{inquiryId}")
    public ResponseEntity<Map<String, String>> updateMyInquiry(
            @RequestParam String userId,
            @PathVariable Integer inquiryId,
            @RequestBody InquiryRequestDto requestDto
    ) {
        inquiryService.updateMyInquiry(userId, inquiryId, requestDto);
        return ResponseEntity.ok(Map.of("message", "문의가 수정되었습니다."));
    }

    /** 내 문의 삭제 */
    @DeleteMapping("/{inquiryId}")
    public ResponseEntity<Map<String, String>> deleteMyInquiry(
            @RequestParam String userId,
            @PathVariable Integer inquiryId
    ) {
        inquiryService.deleteMyInquiry(userId, inquiryId);
        return ResponseEntity.ok(Map.of("message", "문의가 삭제되었습니다."));
    }
}
