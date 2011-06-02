#ifndef GTK_TEXTBUFFER_HPP
#define GTK_TEXTBUFFER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextBuffer
 */
class TextBuffer
    :
    public Gtk::CoreGObject
{
public:

    TextBuffer( const Falcon::CoreClass*, const GtkTextBuffer* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC get_line_count( VMARG );

    static FALCON_FUNC get_char_count( VMARG );

    static FALCON_FUNC get_tag_table( VMARG );

    static FALCON_FUNC insert( VMARG );

    static FALCON_FUNC insert_at_cursor( VMARG );

    static FALCON_FUNC insert_interactive( VMARG );

    static FALCON_FUNC insert_interactive_at_cursor( VMARG );

    static FALCON_FUNC insert_range( VMARG );

    static FALCON_FUNC insert_range_interactive( VMARG );

    //static FALCON_FUNC insert_with_tags( VMARG );

    //static FALCON_FUNC insert_with_tags_by_name( VMARG );

    static FALCON_FUNC delete_( VMARG );

    static FALCON_FUNC delete_interactive( VMARG );

    static FALCON_FUNC backspace( VMARG );

    static FALCON_FUNC set_text( VMARG );

    static FALCON_FUNC get_text( VMARG );

    static FALCON_FUNC get_slice( VMARG );

    static FALCON_FUNC insert_pixbuf( VMARG );

    //static FALCON_FUNC insert_child_anchor( VMARG );

    //static FALCON_FUNC create_child_anchor( VMARG );

    static FALCON_FUNC create_mark( VMARG );

    static FALCON_FUNC move_mark( VMARG );

    static FALCON_FUNC move_mark_by_name( VMARG );

    static FALCON_FUNC add_mark( VMARG );

    static FALCON_FUNC delete_mark( VMARG );

    static FALCON_FUNC delete_mark_by_name( VMARG );

    static FALCON_FUNC get_mark( VMARG );

    static FALCON_FUNC get_insert( VMARG );

    static FALCON_FUNC get_selection_bound( VMARG );

    static FALCON_FUNC get_has_selection( VMARG );

    static FALCON_FUNC place_cursor( VMARG );

    static FALCON_FUNC select_range( VMARG );

    static FALCON_FUNC apply_tag( VMARG );

    static FALCON_FUNC remove_tag( VMARG );

    static FALCON_FUNC apply_tag_by_name( VMARG );

    static FALCON_FUNC remove_tag_by_name( VMARG );

    static FALCON_FUNC remove_all_tags( VMARG );

    //static FALCON_FUNC create_tag( VMARG );

    static FALCON_FUNC get_iter_at_line_offset( VMARG );

    static FALCON_FUNC get_iter_at_offset( VMARG );

    static FALCON_FUNC get_iter_at_line( VMARG );

    static FALCON_FUNC get_iter_at_line_index( VMARG );

    //static FALCON_FUNC get_iter_at_mark( VMARG );

    //static FALCON_FUNC get_iter_at_child_anchor( VMARG );

    static FALCON_FUNC get_start_iter( VMARG );

    static FALCON_FUNC get_end_iter( VMARG );

    static FALCON_FUNC get_bounds( VMARG );

    static FALCON_FUNC get_modified( VMARG );

    static FALCON_FUNC set_modified( VMARG );

    static FALCON_FUNC delete_selection( VMARG );

    //static FALCON_FUNC paste_clipboard( VMARG );

    //static FALCON_FUNC copy_clipboard( VMARG );

    //static FALCON_FUNC cut_clipboard( VMARG );

    static FALCON_FUNC get_selection_bounds( VMARG );

    static FALCON_FUNC begin_user_action( VMARG );

    static FALCON_FUNC end_user_action( VMARG );

    //static FALCON_FUNC add_selection_clipboard( VMARG );

    //static FALCON_FUNC remove_selection_clipboard( VMARG );

    //static FALCON_FUNC deserialize( VMARG );

    //static FALCON_FUNC deserialize_get_can_create_tags( VMARG );

    //static FALCON_FUNC deserialize_set_can_create_tags( VMARG );

    //static FALCON_FUNC get_copy_target_list( VMARG );

    //static FALCON_FUNC get_deserialize_formats( VMARG );

    //static FALCON_FUNC get_paste_target_list( VMARG );

    //static FALCON_FUNC get_serialize_formats( VMARG );

    //static FALCON_FUNC register_deserialize_format( VMARG );

    //static FALCON_FUNC register_deserialize_tagset( VMARG );

    //static FALCON_FUNC register_serialize_format( VMARG );

    //static FALCON_FUNC register_serialize_tagset( VMARG );

    //static FALCON_FUNC gtk_text_buffer_serialize( VMARG );

    //static FALCON_FUNC unregister_deserialize_format( VMARG );

    //static FALCON_FUNC unregister_serialize_format( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEXTBUFFER_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
