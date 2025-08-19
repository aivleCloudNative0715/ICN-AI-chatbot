// com/incheonai/chatbotbackend/service/AirportInfoService.java
package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.dto.TemperatureResponseDto;
import com.incheonai.chatbotbackend.dto.external.*;
import com.incheonai.chatbotbackend.repository.secondary.AtmosRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.util.UriComponentsBuilder;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import java.time.format.DateTimeFormatter;

@Slf4j
@Service
public class AirportInfoService {

    private final WebClient webClient;
    private final AtmosRepository atmosRepository;

    @Value("${api.service-key}")
    private String serviceKey;
    @Value("${api.url.parking}")
    private String parkingApiUrl;
    @Value("${api.url.forecast}")
    private String forecastApiUrl;
//    @Value("${api.url.weather}")
//    private String weatherApiUrl;
    @Value("${api.url.flight_arrivals}")
    private String flightArrivalsApiUrl;
    @Value("${api.url.flight_departures}")
    private String flightDeparturesApiUrl;

    public AirportInfoService(WebClient.Builder webClientBuilder, AtmosRepository atmosRepository) {
        this.webClient = webClientBuilder.build();
        this.atmosRepository = atmosRepository;
    }

//    public Mono<TemperatureResponseDto> getLatestTemperature() {
//        return Mono.fromCallable(atmosRepository::findLatestAtmosData)
//                .subscribeOn(Schedulers.boundedElastic())
//                .flatMap(optionalAtmos -> optionalAtmos
//                        .map(Mono::just)
//                        .orElseGet(() -> Mono.error(new RuntimeException("최신 날씨 데이터를 찾을 수 없습니다."))))
//                .map(atmos -> {
//                    // 온도 값 가공
//                    String tempStr = atmos.getTemperature();
//                    if (tempStr == null || tempStr.isEmpty()) {
//                        throw new RuntimeException("온도 데이터가 유효하지 않습니다.");
//                    }
//                    double temperatureValue = Double.parseDouble(tempStr) / 10.0;
//
//                    // ✅ DB에 저장된 tm 필드 값을 그대로 가져옵니다.
//                    String rawTimestamp = atmos.getTime();
//
//                    // ✅ 변환 없이 DTO를 생성하여 반환합니다.
//                    return new TemperatureResponseDto(temperatureValue, rawTimestamp);
//                });
//    }

    /**
     * ✅ 모든 날씨 데이터를 조회하는 메서드로 변경합니다.
     * @return Mono<List<TemperatureResponseDto>>
     */
    public Mono<TemperatureResponseDto> getLatestTemperature() {
        // 1. 날짜 타입에 맞는 간단한 쿼리 메서드를 호출합니다.
        return Mono.fromCallable(atmosRepository::findFirstByOrderByTimeDesc)
                .subscribeOn(Schedulers.boundedElastic())
                .flatMap(optionalAtmos -> optionalAtmos
                        .map(Mono::just)
                        .orElseGet(() -> Mono.error(new RuntimeException("최신 날씨 데이터를 찾을 수 없습니다."))))
                .map(atmos -> {
                    // 온도 값 가공
                    double temperatureValue = Double.parseDouble(atmos.getTemperature()) / 10.0;

                    // 1. DB에서 Instant 객체를 직접 받습니다.
                    Instant dbTime = atmos.getTime();

                    // 2. ✅ 한국 시간 변환 로직을 삭제하고, UTC 기준으로 포맷터를 설정합니다.
                    DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")
                            .withZone(ZoneId.of("UTC"));

                    // 3. ✅ UTC 시간을 기준으로 문자열을 생성합니다.
                    String formattedTimestamp = outputFormatter.format(dbTime);

                    return new TemperatureResponseDto(temperatureValue, formattedTimestamp);
                });
    }

    private <T> Mono<List<T>> fetchApiData(String baseUrl, MultiValueMap<String, String> queryParams, ParameterizedTypeReference<ApiResult<T>> typeReference) {
        queryParams.add("serviceKey", serviceKey);
        queryParams.add("type", "json");

        URI uri = UriComponentsBuilder.fromHttpUrl(baseUrl).queryParams(queryParams).build(false).toUri();

        return webClient.get()
                .uri(uri)
                .retrieve()
                .bodyToMono(typeReference)
                .map(result -> {
                    // 1. 최상위 'response' 객체를 먼저 가져옵니다.
                    if (result != null && result.getResponse() != null) {
                        ApiResponseData<T> responseData = result.getResponse();
                        ApiResponseData.Header header = responseData.getHeader();
                        ApiResponseData.Body<T> body = responseData.getBody();

                        // 2. 'header'와 'body'는 responseData 객체에서 가져옵니다.
                        if (header != null && "00".equals(header.getResultCode())) {
                            if (body != null && body.getItems() != null) {
                                return body.getItems();
                            }
                        } else if (header != null) {
                            log.error("API returned an error from [{}]: Code={}, Msg={}", baseUrl, header.getResultCode(), header.getResultMsg());
                        }
                    }
                    log.warn("API response from [{}] is empty or has an unexpected structure.", baseUrl);
                    return Collections.<T>emptyList();
                })
                .doOnError(error -> log.error("API call to {} failed: {}", baseUrl, error.getMessage()))
                .onErrorResume(e -> Mono.empty());
    }

    public Mono<List<ParkingInfoItem>> getParkingInfo() {
        return fetchApiData(parkingApiUrl, new LinkedMultiValueMap<>(), new ParameterizedTypeReference<>() {});
    }

    public Mono<List<PassengerForecastItem>> getPassengerForecast(String selectDate) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        if ("0".equals(selectDate) || "1".equals(selectDate)) {
            params.add("selectdate", selectDate);
        }
        return fetchApiData(forecastApiUrl, params, new ParameterizedTypeReference<>() {});
    }

//    public Mono<List<ArrivalWeatherInfoItem>> getArrivalsWeatherInfo() {
//        return fetchApiData(weatherApiUrl, new LinkedMultiValueMap<>(), new ParameterizedTypeReference<>() {});
//    }

    public Mono<List<FlightArrivalInfoItem>> getFlightArrivalsInfo(String flightId) {
        // 공통 파라미터 생성 로직을 호출합니다.
        MultiValueMap<String, String> params = createApiParams(flightId);
        return fetchApiData(flightArrivalsApiUrl, params, new ParameterizedTypeReference<>() {});
    }

    public Mono<List<FlightDepartureInfoItem>> getFlightDeparturesInfo(String flightId) {
        // 공통 파라미터 생성 로직을 호출합니다.
        MultiValueMap<String, String> params = createApiParams(flightId);
        return fetchApiData(flightDeparturesApiUrl, params, new ParameterizedTypeReference<>() {});
    }

    /**
     * API 요청 파라미터를 생성하는 private 헬퍼 메서드입니다.
     * flightId가 없으면 한국 시간(KST) 기준으로 현재 날짜와 시간을 파라미터에 추가합니다.
     * @param flightId 항공편 ID
     * @return API 요청에 사용될 MultiValueMap
     */
    private MultiValueMap<String, String> createApiParams(String flightId) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();

        if (flightId == null || flightId.isEmpty()) {
            // 1. 한국 시간대(KST)를 명시적으로 지정합니다.
            ZoneId kstZoneId = ZoneId.of("Asia/Seoul");

            // 2. 현재 시간을 한국 시간대 기준으로 가져옵니다.
            ZonedDateTime nowInKst = ZonedDateTime.now(kstZoneId);

            // 3. 포맷터를 사용하여 날짜와 시간을 문자열로 변환합니다.
            String todayStr = nowInKst.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            String currentTime = nowInKst.format(DateTimeFormatter.ofPattern("HHmm"));

            params.add("searchday", todayStr);
            params.add("from_time", currentTime);
            params.add("to_time", "2359");
        } else {
            params.add("flight_id", flightId);
        }
        return params;
    }

//    public Mono<List<FlightArrivalInfoItem>> getFlightArrivalsInfo(String flightId) {
//        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
//
//        if (flightId == null || flightId.isEmpty()) {
//            // 오늘 날짜를 "YYYYMMDD" 형식으로 만듭니다.
//            String todayStr = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
//            String currentTime = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmm"));
//
//            // searchday 파라미터를 추가하여 조회 날짜를 오늘로 고정합니다.
//            params.add("searchday", todayStr);
//            params.add("from_time", currentTime);
//            params.add("to_time", "2359");
//        } else {
//            params.add("flight_id", flightId);
//        }
//
//        return fetchApiData(flightArrivalsApiUrl, params, new ParameterizedTypeReference<>() {});
//    }
//
//    public Mono<List<FlightDepartureInfoItem>> getFlightDeparturesInfo(String flightId) {
//        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
//
//        if (flightId == null || flightId.isEmpty()) {
//            String todayStr = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
//            String currentTime = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmm"));
//
//            // searchday 파라미터를 추가하여 조회 날짜를 오늘로 고정합니다.
//            params.add("searchday", todayStr);
//            params.add("from_time", currentTime);
//            params.add("to_time", "2359");
//        } else {
//            params.add("flight_id", flightId);
//        }
//
//        return fetchApiData(flightDeparturesApiUrl, params, new ParameterizedTypeReference<>() {});
//    }
}