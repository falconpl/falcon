#ifndef GTK_TEXTITER_HPP
#define GTK_TEXTITER_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextIter
 */
class TextIter
    :
    public Falcon::CoreObject
{
public:

    TextIter( const Falcon::CoreClass*, const GtkTextIter* = 0 );

    ~TextIter();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC get_buffer( VMARG );

    static FALCON_FUNC copy( VMARG );

    //static FALCON_FUNC free( VMARG );

    static FALCON_FUNC get_offset( VMARG );

    static FALCON_FUNC get_line( VMARG );

    static FALCON_FUNC get_line_offset( VMARG );

    static FALCON_FUNC get_line_index( VMARG );

    static FALCON_FUNC get_visible_line_index( VMARG );

    static FALCON_FUNC get_visible_line_offset( VMARG );

    static FALCON_FUNC get_char( VMARG );

    static FALCON_FUNC get_slice( VMARG );

    static FALCON_FUNC get_text( VMARG );

    static FALCON_FUNC get_visible_slice( VMARG );

    static FALCON_FUNC get_visible_text( VMARG );

    static FALCON_FUNC get_pixbuf( VMARG );
#if 0
    static FALCON_FUNC get_marks( VMARG );

    static FALCON_FUNC get_toggled_tags( VMARG );

    static FALCON_FUNC get_child_anchor( VMARG );

    static FALCON_FUNC begins_tag( VMARG );

    static FALCON_FUNC ends_tag( VMARG );

    static FALCON_FUNC toggles_tag( VMARG );

    static FALCON_FUNC has_tag( VMARG );

    static FALCON_FUNC get_tags( VMARG );

    static FALCON_FUNC editable( VMARG );

    static FALCON_FUNC can_insert( VMARG );

    static FALCON_FUNC starts_word( VMARG );

    static FALCON_FUNC ends_word( VMARG );

    static FALCON_FUNC inside_word( VMARG );

    static FALCON_FUNC starts_line( VMARG );

    static FALCON_FUNC ends_line( VMARG );

    static FALCON_FUNC starts_sentence( VMARG );

    static FALCON_FUNC ends_sentence( VMARG );

    static FALCON_FUNC inside_sentence( VMARG );

    static FALCON_FUNC is_cursor_position( VMARG );

    static FALCON_FUNC get_chars_in_line( VMARG );

    static FALCON_FUNC get_bytes_in_line( VMARG );

    static FALCON_FUNC get_attributes( VMARG );

    static FALCON_FUNC get_language( VMARG );

    static FALCON_FUNC is_end( VMARG );

    static FALCON_FUNC is_start( VMARG );

    static FALCON_FUNC forward_char( VMARG );

    static FALCON_FUNC backward_char( VMARG );

    static FALCON_FUNC forward_chars( VMARG );

    static FALCON_FUNC backward_chars( VMARG );

    static FALCON_FUNC forward_line( VMARG );

    static FALCON_FUNC backward_line( VMARG );

    static FALCON_FUNC forward_lines( VMARG );

    static FALCON_FUNC backward_lines( VMARG );

    static FALCON_FUNC forward_word_ends( VMARG );

    static FALCON_FUNC backward_word_starts( VMARG );

    static FALCON_FUNC forward_word_end( VMARG );

    static FALCON_FUNC backward_word_start( VMARG );

    static FALCON_FUNC forward_cursor_position( VMARG );

    static FALCON_FUNC backward_cursor_position( VMARG );

    static FALCON_FUNC forward_cursor_positions( VMARG );

    static FALCON_FUNC backward_cursor_positions( VMARG );

    static FALCON_FUNC backward_sentence_start( VMARG );

    static FALCON_FUNC backward_sentence_starts( VMARG );

    static FALCON_FUNC forward_sentence_end( VMARG );

    static FALCON_FUNC forward_sentence_ends( VMARG );

    static FALCON_FUNC forward_visible_word_ends( VMARG );

    static FALCON_FUNC backward_visible_word_starts( VMARG );

    static FALCON_FUNC forward_visible_word_end( VMARG );

    static FALCON_FUNC backward_visible_word_start( VMARG );

    static FALCON_FUNC forward_visible_cursor_position( VMARG );

    static FALCON_FUNC backward_visible_cursor_position( VMARG );

    static FALCON_FUNC forward_visible_cursor_positions( VMARG );

    static FALCON_FUNC backward_visible_cursor_positions( VMARG );

    static FALCON_FUNC forward_visible_line( VMARG );

    static FALCON_FUNC backward_visible_line( VMARG );

    static FALCON_FUNC forward_visible_lines( VMARG );

    static FALCON_FUNC backward_visible_lines( VMARG );

    static FALCON_FUNC set_offset( VMARG );

    static FALCON_FUNC set_line( VMARG );

    static FALCON_FUNC set_line_offset( VMARG );

    static FALCON_FUNC set_line_index( VMARG );

    static FALCON_FUNC set_visible_line_index( VMARG );

    static FALCON_FUNC set_visible_line_offset( VMARG );

    static FALCON_FUNC forward_to_end( VMARG );

    static FALCON_FUNC forward_to_line_end( VMARG );

    static FALCON_FUNC forward_to_tag_toggle( VMARG );

    static FALCON_FUNC backward_to_tag_toggle( VMARG );

    static FALCON_FUNC forward_find_char( VMARG );

    static FALCON_FUNC backward_find_char( VMARG );

    static FALCON_FUNC forward_search( VMARG );

    static FALCON_FUNC backward_search( VMARG );

    static FALCON_FUNC equal( VMARG );

    static FALCON_FUNC compare( VMARG );

    static FALCON_FUNC in_range( VMARG );

    static FALCON_FUNC order( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_TEXTITER_HPP
