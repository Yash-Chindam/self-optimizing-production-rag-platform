{{- define "rag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "rag.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
app.kubernetes.io/name: {{ include "rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "rag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "rag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "rag.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}

{{- define "rag.secretName" -}}
{{- default (printf "%s-secrets" (include "rag.fullname" .)) .Values.secrets.existingSecret -}}
{{- end -}}

{{/* True when the named in-cluster backing service is deployed. */}}
{{- define "rag.backing" -}}
{{- $service := index .root.Values.backingServices.services .name -}}
{{- if and .root.Values.backingServices.enabled $service $service.enabled -}}true{{- end -}}
{{- end -}}

{{/*
An endpoint: the explicit value if set, else the in-cluster backing service, else empty.
Call with (dict "root" . "value" <explicit> "name" <service> "format" <printf with one %s>).
*/}}
{{- define "rag.endpoint" -}}
{{- if .value -}}
{{- .value -}}
{{- else if include "rag.backing" (dict "root" .root "name" .name) -}}
{{- printf .format (printf "%s-%s" (include "rag.fullname" .root) .name) -}}
{{- end -}}
{{- end -}}

{{- define "rag.otlpEndpoint" -}}
{{- if .Values.endpoints.otlpEndpoint -}}
{{- .Values.endpoints.otlpEndpoint -}}
{{- else if .Values.otelCollector.enabled -}}
{{- printf "http://%s-otel-collector:4318" (include "rag.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* The environment every workload of the application shares. */}}
{{- define "rag.envFrom" -}}
- configMapRef:
    name: {{ include "rag.fullname" . }}-config
- secretRef:
    name: {{ include "rag.secretName" . }}
    optional: true
{{- end -}}

{{/* Pod-level settings shared by the API, the evaluation job and the Prefect worker. */}}
{{- define "rag.podSpec" -}}
serviceAccountName: {{ include "rag.serviceAccountName" . }}
automountServiceAccountToken: false
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
