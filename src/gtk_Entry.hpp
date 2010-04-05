#ifndef GTK_ENTRY_HPP
#define GTK_ENTRY_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Entry
 */
class Entry
    :
    public Gtk::CoreGObject
{
public:

    Entry( const Falcon::CoreClass*, const GtkEntry* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if GTK_MINOR_VERSION >= 18

    static FALCON_FUNC get_buffer( VMARG );

    static FALCON_FUNC set_buffer( VMARG );

#endif

    static FALCON_FUNC set_text( VMARG );

    //static FALCON_FUNC append_text( VMARG );

    //static FALCON_FUNC prepend_text( VMARG );

    //static FALCON_FUNC set_position( VMARG );

    static FALCON_FUNC get_text( VMARG );

    static FALCON_FUNC get_text_length( VMARG );

    //static FALCON_FUNC select_region( VMARG );

    static FALCON_FUNC set_visibility( VMARG );

    static FALCON_FUNC set_invisible_char( VMARG );

#if GTK_MINOR_VERSION >= 16
    static FALCON_FUNC unset_invisible_char( VMARG );
#endif

    //static FALCON_FUNC set_editable( VMARG );

    static FALCON_FUNC set_max_length( VMARG );

    static FALCON_FUNC get_activates_default( VMARG );

    //static FALCON_FUNC get_has_frame( VMARG );

    //static FALCON_FUNC get_inner_border( VMARG );

    //static FALCON_FUNC get_width_chars( VMARG );

    //static FALCON_FUNC set_activates_default( VMARG );

    //static FALCON_FUNC set_has_frame( VMARG );

    //static FALCON_FUNC set_inner_border( VMARG );

    //static FALCON_FUNC set_width_chars( VMARG );

    //static FALCON_FUNC get_invisible_char( VMARG );

    //static FALCON_FUNC set_alignment( VMARG );

    //static FALCON_FUNC get_alignment( VMARG );

    //static FALCON_FUNC set_overwrite_mode( VMARG );

    //static FALCON_FUNC get_overwrite_mode( VMARG );

    //static FALCON_FUNC get_layout( VMARG );

    //static FALCON_FUNC get_layout_offsets( VMARG );

    //static FALCON_FUNC layout_index_to_text_index( VMARG );

    //static FALCON_FUNC text_index_to_layout_index( VMARG );

    //static FALCON_FUNC get_max_length( VMARG );

    //static FALCON_FUNC get_visibility( VMARG );

    //static FALCON_FUNC set_completion( VMARG );

    //static FALCON_FUNC get_completion( VMARG );

    //static FALCON_FUNC set_cursor_hadjustment( VMARG );

    //static FALCON_FUNC get_cursor_hadjustment( VMARG );

    //static FALCON_FUNC set_progress_fraction( VMARG );

    //static FALCON_FUNC get_progress_fraction( VMARG );

    //static FALCON_FUNC set_progress_pulse_step( VMARG );

    //static FALCON_FUNC get_progress_pulse_step( VMARG );

    //static FALCON_FUNC progress_pulse( VMARG );

    //static FALCON_FUNC set_icon_from_pixbuf( VMARG );

    //static FALCON_FUNC set_icon_from_stock( VMARG );

    //static FALCON_FUNC set_icon_from_icon_name( VMARG );

    //static FALCON_FUNC set_icon_from_gicon( VMARG );

    //static FALCON_FUNC get_icon_storage_type( VMARG );

    //static FALCON_FUNC get_icon_pixbuf( VMARG );

    //static FALCON_FUNC get_icon_stock( VMARG );

    //static FALCON_FUNC get_icon_name( VMARG );

    //static FALCON_FUNC get_icon_gicon( VMARG );

    //static FALCON_FUNC set_icon_activatable( VMARG );

    //static FALCON_FUNC get_icon_activatable( VMARG );

    //static FALCON_FUNC set_icon_sensitive( VMARG );

    //static FALCON_FUNC get_icon_sensitive( VMARG );

    //static FALCON_FUNC get_icon_at_pos( VMARG );

    //static FALCON_FUNC set_icon_tooltip_text( VMARG );

    //static FALCON_FUNC get_icon_tooltip_text( VMARG );

    //static FALCON_FUNC set_icon_tooltip_markup( VMARG );

    //static FALCON_FUNC get_icon_tooltip_markup( VMARG );

    //static FALCON_FUNC set_icon_drag_source( VMARG );

    //static FALCON_FUNC get_current_icon_drag_source( VMARG );

    //static FALCON_FUNC get_icon_window( VMARG );

    //static FALCON_FUNC get_text_window( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ENTRY_HPP
