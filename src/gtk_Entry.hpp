#ifndef GTK_ENTRY_HPP
#define GTK_ENTRY_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Entry
 */
namespace Entry {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

#if GTK_MINOR_VERSION >= 18

FALCON_FUNC get_buffer( VMARG );

FALCON_FUNC set_buffer( VMARG );

#endif

FALCON_FUNC set_text( VMARG );

//FALCON_FUNC append_text( VMARG );

//FALCON_FUNC prepend_text( VMARG );

//FALCON_FUNC set_position( VMARG );

FALCON_FUNC get_text( VMARG );

FALCON_FUNC get_text_length( VMARG );

//FALCON_FUNC select_region( VMARG );

FALCON_FUNC set_visibility( VMARG );

FALCON_FUNC set_invisible_char( VMARG );

#if GTK_MINOR_VERSION >= 16
FALCON_FUNC unset_invisible_char( VMARG );
#endif

//FALCON_FUNC set_editable( VMARG );

FALCON_FUNC set_max_length( VMARG );

FALCON_FUNC get_activates_default( VMARG );

//FALCON_FUNC get_has_frame( VMARG );

//FALCON_FUNC get_inner_border( VMARG );

//FALCON_FUNC get_width_chars( VMARG );

//FALCON_FUNC set_activates_default( VMARG );

//FALCON_FUNC set_has_frame( VMARG );

//FALCON_FUNC set_inner_border( VMARG );

//FALCON_FUNC set_width_chars( VMARG );

//FALCON_FUNC get_invisible_char( VMARG );

//FALCON_FUNC set_alignment( VMARG );

//FALCON_FUNC get_alignment( VMARG );

//FALCON_FUNC set_overwrite_mode( VMARG );

//FALCON_FUNC get_overwrite_mode( VMARG );

//FALCON_FUNC get_layout( VMARG );

//FALCON_FUNC get_layout_offsets( VMARG );

//FALCON_FUNC layout_index_to_text_index( VMARG );

//FALCON_FUNC text_index_to_layout_index( VMARG );

//FALCON_FUNC get_max_length( VMARG );

//FALCON_FUNC get_visibility( VMARG );

//FALCON_FUNC set_completion( VMARG );

//FALCON_FUNC get_completion( VMARG );

//FALCON_FUNC set_cursor_hadjustment( VMARG );

//FALCON_FUNC get_cursor_hadjustment( VMARG );

//FALCON_FUNC set_progress_fraction( VMARG );

//FALCON_FUNC get_progress_fraction( VMARG );

//FALCON_FUNC set_progress_pulse_step( VMARG );

//FALCON_FUNC get_progress_pulse_step( VMARG );

//FALCON_FUNC progress_pulse( VMARG );

//FALCON_FUNC set_icon_from_pixbuf( VMARG );

//FALCON_FUNC set_icon_from_stock( VMARG );

//FALCON_FUNC set_icon_from_icon_name( VMARG );

//FALCON_FUNC set_icon_from_gicon( VMARG );

//FALCON_FUNC get_icon_storage_type( VMARG );

//FALCON_FUNC get_icon_pixbuf( VMARG );

//FALCON_FUNC get_icon_stock( VMARG );

//FALCON_FUNC get_icon_name( VMARG );

//FALCON_FUNC get_icon_gicon( VMARG );

//FALCON_FUNC set_icon_activatable( VMARG );

//FALCON_FUNC get_icon_activatable( VMARG );

//FALCON_FUNC set_icon_sensitive( VMARG );

//FALCON_FUNC get_icon_sensitive( VMARG );

//FALCON_FUNC get_icon_at_pos( VMARG );

//FALCON_FUNC set_icon_tooltip_text( VMARG );

//FALCON_FUNC get_icon_tooltip_text( VMARG );

//FALCON_FUNC set_icon_tooltip_markup( VMARG );

//FALCON_FUNC get_icon_tooltip_markup( VMARG );

//FALCON_FUNC set_icon_drag_source( VMARG );

//FALCON_FUNC get_current_icon_drag_source( VMARG );

//FALCON_FUNC get_icon_window( VMARG );

//FALCON_FUNC get_text_window( VMARG );


} // Entry
} // Gtk
} // Falcon

#endif // !GTK_ENTRY_HPP
