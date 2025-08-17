// com/incheonai/chatbotbackend/service/AirportInfoService.java
package com.incheonai.chatbotbackend.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import com.incheonai.chatbotbackend.dto.external.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.util.UriComponentsBuilder;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
public class AirportInfoService {

    private final WebClient webClient;
    private final XmlMapper xmlMapper;

    @Value("${api.service-key}")
    private String serviceKey;

    @Value("${api.url.parking}")
    private String parkingApiUrl;

    @Value("${api.url.weather}")
    private String weatherApiUrl;

    @Value("${api.url.flight_arrivals}")
    private String flightArrivalsApiUrl;

    @Value("${api.url.flight_departures}")
    private String flightDeparturesApiUrl;

    @Value("${api.url.forecast}")
    private String forecastApiUrl;

    public AirportInfoService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
        this.xmlMapper = new XmlMapper();
        this.xmlMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    private <T> Mono<List<T>> fetchApiData(String baseUrl, MultiValueMap<String, String> queryParams, TypeReference<ApiResult<T>> typeReference) {
        queryParams.add("serviceKey", serviceKey);

        URI uri = UriComponentsBuilder
                .fromHttpUrl(baseUrl)
                .queryParams(queryParams) // 파라미터 추가
                .build(false)
                .toUri();

        return webClient.get()
                .uri(uri)
                .retrieve()
                .bodyToMono(String.class)
                .flatMap(xmlString -> {
                    // =================================================================
                    // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 줄을 추가합니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                    log.info("Raw XML Response from [{}]: {}", baseUrl, xmlString);
                    // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 이 줄을 추가합니다 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                    // =================================================================
                    try {
                        ApiResult<T> result = xmlMapper.readValue(xmlString, typeReference);

                        if (result.getBody() != null && result.getBody().getItems() != null && result.getBody().getItems().getItem() != null) {
                            return Mono.just(result.getBody().getItems().getItem());
                        }

                        if (result.getHeader() != null && result.getHeader().getResultCode() != null) {
                            log.error("API returned an error: Code={}, Msg={}", result.getHeader().getResultCode(), result.getHeader().getResultMsg());
                        } else {
                            log.warn("API response is empty or has an unexpected structure for URL: {}", baseUrl);
                        }

                        return Mono.just(Collections.<T>emptyList());

                    } catch (Exception e) {
                        log.error("XML parsing failed for URL: {}. XML content: {}", baseUrl, xmlString, e);
                        return Mono.just(Collections.<T>emptyList());
                    }
                })
                .doOnError(error -> log.error("API call to {} failed: {}", baseUrl, error.getMessage()))
                .onErrorResume(e -> Mono.<List<T>>empty());
    }

    public Mono<List<ParkingInfoItem>> getParkingInfo() {
        return fetchApiData(parkingApiUrl, new LinkedMultiValueMap<>(), new TypeReference<>() {});
    }

    public Mono<List<ArrivalWeatherInfoItem>> getArrivalsWeatherInfo() {
        return fetchApiData(weatherApiUrl, new LinkedMultiValueMap<>(), new TypeReference<>() {});
    }

    // 항공편 도착 정보 조회 메서드 추가
    public Mono<List<FlightArrivalInfoItem>> getFlightArrivalsInfo(String flightId) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        if (flightId != null && !flightId.isEmpty()) {
            params.add("flight_id", flightId);
        }
        return fetchApiData(flightArrivalsApiUrl, params, new TypeReference<>() {});
    }

    // 항공편 출발 정보 조회 메서드 추가
    public Mono<List<FlightDepartureInfoItem>> getFlightDeparturesInfo(String flightId) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        if (flightId != null && !flightId.isEmpty()) {
            params.add("flight_id", flightId);
        }
        return fetchApiData(flightDeparturesApiUrl, params, new TypeReference<>() {});
    }

    public Mono<List<PassengerForecastItem>> getPassengerForecast() {
        return fetchApiData(forecastApiUrl, new LinkedMultiValueMap<>(), new TypeReference<>() {});
    }
}