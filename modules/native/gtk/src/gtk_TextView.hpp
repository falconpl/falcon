#ifndef GTK_TEXTVIEW_HPP
#define GTK_TEXTVIEW_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextView
 */
class TextView
    :
    public Gtk::CoreGObject
{
public:

    TextView( const Falcon::CoreClass*, const GtkTextView* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_buffer( VMARG );

    static FALCON_FUNC set_buffer( VMARG );

    static FALCON_FUNC get_buffer( VMARG );

    static FALCON_FUNC scroll_to_mark( VMARG );

    static FALCON_FUNC scroll_to_iter( VMARG );

    static FALCON_FUNC scroll_mark_onscreen( VMARG );

    static FALCON_FUNC move_mark_onscreen( VMARG );

    static FALCON_FUNC place_cursor_onscreen( VMARG );

    //static FALCON_FUNC get_visible_rect( VMARG );

    //static FALCON_FUNC get_iter_location( VMARG );

    static FALCON_FUNC get_line_at_y( VMARG );

    static FALCON_FUNC get_line_yrange( VMARG );

    static FALCON_FUNC get_iter_at_location( VMARG );

    static FALCON_FUNC get_iter_at_position( VMARG );

    static FALCON_FUNC buffer_to_window_coords( VMARG );

    static FALCON_FUNC window_to_buffer_coords( VMARG );

    //static FALCON_FUNC get_window( VMARG );

    //static FALCON_FUNC get_window_type( VMARG );

    static FALCON_FUNC set_border_window_size( VMARG );

    static FALCON_FUNC get_border_window_size( VMARG );

    static FALCON_FUNC forward_display_line( VMARG );

    static FALCON_FUNC backward_display_line( VMARG );

    static FALCON_FUNC forward_display_line_end( VMARG );

    static FALCON_FUNC backward_display_line_start( VMARG );

    static FALCON_FUNC starts_display_line( VMARG );

    static FALCON_FUNC move_visually( VMARG );

    //static FALCON_FUNC add_child_at_anchor( VMARG );

    static FALCON_FUNC add_child_in_window( VMARG );

    static FALCON_FUNC move_child( VMARG );

    static FALCON_FUNC set_wrap_mode( VMARG );

    static FALCON_FUNC get_wrap_mode( VMARG );

    static FALCON_FUNC set_editable( VMARG );

    static FALCON_FUNC get_editable( VMARG );

    static FALCON_FUNC set_cursor_visible( VMARG );

    static FALCON_FUNC get_cursor_visible( VMARG );

    static FALCON_FUNC set_overwrite( VMARG );

    static FALCON_FUNC get_overwrite( VMARG );

    static FALCON_FUNC set_pixels_above_lines( VMARG );

    static FALCON_FUNC get_pixels_above_lines( VMARG );

    static FALCON_FUNC set_pixels_below_lines( VMARG );

    static FALCON_FUNC get_pixels_below_lines( VMARG );

    static FALCON_FUNC set_pixels_inside_wrap( VMARG );

    static FALCON_FUNC get_pixels_inside_wrap( VMARG );

    static FALCON_FUNC set_justification( VMARG );

    static FALCON_FUNC get_justification( VMARG );

    static FALCON_FUNC set_left_margin( VMARG );

    static FALCON_FUNC get_left_margin( VMARG );

    static FALCON_FUNC set_right_margin( VMARG );

    static FALCON_FUNC get_right_margin( VMARG );

    static FALCON_FUNC set_indent( VMARG );

    static FALCON_FUNC get_indent( VMARG );

    //static FALCON_FUNC set_tabs( VMARG );

    //static FALCON_FUNC get_tabs( VMARG );

    static FALCON_FUNC set_accepts_tab( VMARG );

    static FALCON_FUNC get_accepts_tab( VMARG );

    //static FALCON_FUNC get_default_attributes( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEXTVIEW_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
