"""The admin module of the polygons app."""


from django.contrib import admin
from polygons.models import Boundery, Address


class BounderyAdmin(admin.ModelAdmin):
    """Boundery admin page."""

    save_on_top = True
    list_display = ('id', 'area', 'state', 'polygon', 'matchingId')
    search_fields = ('matchingId',)
    list_filter = ('isMain', 'state', 'processed')
    list_per_page = 20


class AddressAdmin(admin.ModelAdmin):
    """Address admin page."""

    save_on_top = True
    list_display = ('id', 'formattedAddress', 'point', 'confidence')
    search_fields = ('formattedAddress',)
    readonly_fields = ('boundery',)
    list_per_page = 20


admin.site.register(Boundery, BounderyAdmin)
admin.site.register(Address, AddressAdmin)
