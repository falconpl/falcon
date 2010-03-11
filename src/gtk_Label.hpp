#ifndef GTK_LABEL_HPP
#define GTK_LABEL_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Label
 */
namespace Label {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set_text( VMARG );

//FALCON_FUNC set_attributes( VMARG );

FALCON_FUNC set_markup( VMARG );

FALCON_FUNC set_markup_with_mnemonic( VMARG );

FALCON_FUNC set_pattern( VMARG );

//FALCON_FUNC set_justify( VMARG );

//FALCON_FUNC set_ellipsize( VMARG );

FALCON_FUNC set_width_chars( VMARG );

FALCON_FUNC set_max_width_chars( VMARG );

//FALCON_FUNC get( VMARG ) deprecated

//FALCON_FUNC parse_uline( VMARG ) deprecated

FALCON_FUNC set_line_wrap( VMARG );

//FALCON_FUNC set_line_wrap_mode( VMARG );

//FALCON_FUNC set( VMARG ); deprecated

//FALCON_FUNC get_layout_offsets( VMARG );

FALCON_FUNC get_mnemonic_keyval( VMARG );

FALCON_FUNC get_selectable( VMARG );

FALCON_FUNC get_text( VMARG );

//FALCON_FUNC new_with_mnemonic( VMARG );

FALCON_FUNC select_region( VMARG );

FALCON_FUNC set_mnemonic_widget( VMARG );

FALCON_FUNC set_selectable( VMARG );

FALCON_FUNC set_text_with_mnemonic( VMARG );

//FALCON_FUNC get_attributes( VMARG );

//FALCON_FUNC get_justify( VMARG );

//FALCON_FUNC get_ellipsize( VMARG );

FALCON_FUNC get_width_chars( VMARG );

FALCON_FUNC get_max_width_chars( VMARG );

FALCON_FUNC get_label( VMARG );

//FALCON_FUNC get_layout( VMARG );

FALCON_FUNC get_line_wrap( VMARG );

//FALCON_FUNC get_line_wrap_mode( VMARG );

FALCON_FUNC get_mnemonic_widget( VMARG );

//FALCON_FUNC get_selection_bounds( VMARG );

FALCON_FUNC get_use_markup( VMARG );

FALCON_FUNC get_use_underline( VMARG );

FALCON_FUNC get_single_line_mode( VMARG );

FALCON_FUNC get_angle( VMARG );

FALCON_FUNC set_label( VMARG );

FALCON_FUNC set_use_markup( VMARG );

FALCON_FUNC set_use_underline( VMARG );

FALCON_FUNC set_single_line_mode( VMARG );

FALCON_FUNC set_angle( VMARG );

#if GTK_VERSION_MINOR >= 18

FALCON_FUNC get_current_uri( VMARG );

FALCON_FUNC set_track_visited_links( VMARG );

FALCON_FUNC get_track_visited_links( VMARG );

#endif


} // Label
} // Gtk
} // Falcon

#endif // !GTK_LABEL_HPP
