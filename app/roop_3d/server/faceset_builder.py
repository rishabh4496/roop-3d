"""Media builder jobs, archive listing, and curation publication."""
from fastapi import APIRouter, Request
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool
from roop_3d.core.settings import Contract
from roop_3d.faceset_builder import FacesetBuilder

router = APIRouter(prefix='/api/v1/faceset')

class ExtractMedia(Contract):
    media_path: str | None = Field(default=None, max_length=1024)
    image_paths: list[str] = Field(default_factory=list, max_length=1000)
    faceset_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')
    max_frames: int = Field(default=300, ge=1, le=3000)
    frame_stride: int = Field(default=5, ge=1, le=300)

    @model_validator(mode='after')
    def inputs(self):
        if not self.media_path and not self.image_paths and not self.faceset_id:
            raise ValueError('Provide media_path, uploaded image_paths, or faceset_id to edit')
        return self

class SaveFaceset(Contract):
    job_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    cluster_rank: int = Field(ge=0, le=15)
    selected_ids: list[str] = Field(min_length=1, max_length=96)
    name: str = Field(min_length=1, max_length=128)
    faceset_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')


class SynthesizeFaceset(Contract):
    job_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    cluster_rank: int = Field(ge=0, le=15)
    selected_ids: list[str] = Field(min_length=1, max_length=3)
    name: str = Field(min_length=1, max_length=128)


class SynthesizePhotos(Contract):
    images: list[str] = Field(min_length=1, max_length=3)
    name: str = Field(min_length=1, max_length=128)


@router.post('/synthesize-photos')
def synthesize_photos(value: SynthesizePhotos, request: Request):
    from roop_3d.faceset_synthesizer import start_photo_synthesis
    return start_photo_synthesis(service(request), value.images, value.name)


@router.post('/synthesize-36')
def synthesize(value: SynthesizeFaceset, request: Request):
    from roop_3d.faceset_synthesizer import start_synthesis
    return start_synthesis(service(request), value.job_id, value.cluster_rank, value.selected_ids, value.name)


def service(request):
    return request.app.state.faceset_builder

@router.post('/extract-from-media')
async def extract(request: Request):
    from roop_3d.core.errors import DomainError
    from pydantic import ValidationError
    if request.headers.get('content-type', '').startswith('multipart/form-data'):
        from pathlib import Path
        import uuid
        paths = []
        try:
            async with request.form(max_files=256, max_fields=10) as form:
                for _, upload in form.multi_items():
                    if not hasattr(upload, 'read'):
                        continue
                    suffix = Path(upload.filename or '').suffix.lower()
                    if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.mov', '.mkv', '.avi', '.webm'}:
                        raise DomainError('MEDIA_FORMAT_INVALID', 'Upload images or reference videos', 422)
                    path = request.app.state.root / 'roop_3d/.runtime/media' / (uuid.uuid4().hex + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    paths.append(str(path.relative_to(request.app.state.root)))
                    total = 0
                    with path.open('wb') as stream:
                        while chunk := await upload.read(1024 * 1024):
                            total += len(chunk)
                            if total > 512 * 1024 * 1024:
                                raise DomainError('MEDIA_TOO_LARGE', 'Use chunked asset uploads for files over 512 MiB', 413)
                            await run_in_threadpool(stream.write, chunk)
                value = ExtractMedia(image_paths=paths, faceset_id=form.get('faceset_id') or None)
        except Exception as error:
            for relative in paths:
                (request.app.state.root / relative).unlink(missing_ok=True)
            if isinstance(error, ValidationError):
                raise DomainError('VALIDATION_ERROR', 'Provide reference files or an existing faceset_id', 422) from None
            raise
    else:
        try:
            value = ExtractMedia.model_validate(await request.json())
        except (ValidationError, ValueError):
            raise DomainError('VALIDATION_ERROR', 'Provide valid media paths and extraction options', 422) from None
    return service(request).start(([value.media_path] if value.media_path else []) + value.image_paths,
        faceset_id=value.faceset_id, max_frames=value.max_frames, frame_stride=value.frame_stride)

@router.get('/extraction/{job_id}')
def extraction(job_id: str, request: Request):
    return service(request).snapshot(job_id)

@router.post('/save-fsz')
async def save(value: SaveFaceset, request: Request):
    return await run_in_threadpool(service(request).save, value.job_id, value.cluster_rank,
                                  value.selected_ids, value.name, value.faceset_id)

@router.get('/list')
def listing(request: Request):
    import json
    root = request.app.state.root
    result = []
    for path in sorted((root / 'facesets').glob('*.fsz')):
        import zipfile
        from roop_3d.faceset.validator import ArchiveValidator
        from roop_3d.core.errors import DomainError
        try:
            with zipfile.ZipFile(path) as archive:
                ArchiveValidator.validate_zip(archive)
                metadata = json.loads(archive.read('metadata.json')) if 'metadata.json' in archive.namelist() else {}
                points = json.loads(archive.read('sphere_coverage.json')) if 'sphere_coverage.json' in archive.namelist() else None
                count = len(metadata['sources']) if 'sources' in metadata else sum(
                    name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and name != 'uv_albedo.png' for name in archive.namelist())
            manifest = root / 'roop_3d/.runtime/facesets' / (path.stem + '.json')
            name = json.loads(manifest.read_text('utf-8')).get('name', metadata.get('source_name', path.stem)) if manifest.is_file() else metadata.get('source_name', path.stem)
            result.append(dict(faceset_id=path.stem, name=name, path='facesets/' + path.name,
                               item_count=count, sphere_coverage_count=len({p['bin_index'] for p in points}) if points is not None else None))
        except (zipfile.BadZipFile, DomainError, ValueError, KeyError, TypeError, AttributeError):
            result.append(dict(faceset_id=path.stem, name=path.stem, path='facesets/' + path.name, status='invalid'))
    return result
