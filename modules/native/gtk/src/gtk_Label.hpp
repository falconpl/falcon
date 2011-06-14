#ifndef GTK_LABEL_HPP
#define GTK_LABEL_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Label
 */
class Label
    :
    public Gtk::CoreGObject
{
public:

    Label( const Falcon::CoreClass*, const GtkLabel* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_text( VMARG );

    //static FALCON_FUNC set_attributes( VMARG );

    static FALCON_FUNC set_markup( VMARG );

    static FALCON_FUNC set_markup_with_mnemonic( VMARG );

    static FALCON_FUNC set_pattern( VMARG );

    //static FALCON_FUNC set_justify( VMARG );

    //static FALCON_FUNC set_ellipsize( VMARG );

    static FALCON_FUNC set_width_chars( VMARG );

    static FALCON_FUNC set_max_width_chars( VMARG );

    //static FALCON_FUNC get( VMARG ) deprecated

    //static FALCON_FUNC parse_uline( VMARG ) deprecated

    static FALCON_FUNC set_line_wrap( VMARG );

    //static FALCON_FUNC set_line_wrap_mode( VMARG );

    //static FALCON_FUNC set( VMARG ); deprecated

    //static FALCON_FUNC get_layout_offsets( VMARG );

    static FALCON_FUNC get_mnemonic_keyval( VMARG );

    static FALCON_FUNC get_selectable( VMARG );

    static FALCON_FUNC get_text( VMARG );

    //static FALCON_FUNC new_with_mnemonic( VMARG );

    static FALCON_FUNC select_region( VMARG );

    static FALCON_FUNC set_mnemonic_widget( VMARG );

    static FALCON_FUNC set_selectable( VMARG );

    static FALCON_FUNC set_text_with_mnemonic( VMARG );

    //static FALCON_FUNC get_attributes( VMARG );

    //static FALCON_FUNC get_justify( VMARG );

    //static FALCON_FUNC get_ellipsize( VMARG );

    static FALCON_FUNC get_width_chars( VMARG );

    static FALCON_FUNC get_max_width_chars( VMARG );

    static FALCON_FUNC get_label( VMARG );

    //static FALCON_FUNC get_layout( VMARG );

    static FALCON_FUNC get_line_wrap( VMARG );

    //static FALCON_FUNC get_line_wrap_mode( VMARG );

    static FALCON_FUNC get_mnemonic_widget( VMARG );

    //static FALCON_FUNC get_selection_bounds( VMARG );

    static FALCON_FUNC get_use_markup( VMARG );

    static FALCON_FUNC get_use_underline( VMARG );

    static FALCON_FUNC get_single_line_mode( VMARG );

    static FALCON_FUNC get_angle( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC set_use_markup( VMARG );

    static FALCON_FUNC set_use_underline( VMARG );

    static FALCON_FUNC set_single_line_mode( VMARG );

    static FALCON_FUNC set_angle( VMARG );

    #if GTK_VERSION_MINOR >= 18

    static FALCON_FUNC get_current_uri( VMARG );

    static FALCON_FUNC set_track_visited_links( VMARG );

    static FALCON_FUNC get_track_visited_links( VMARG );

    #endif

};


} // Gtk
} // Falcon

#endif // !GTK_LABEL_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
