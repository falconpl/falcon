
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         flc_src_parse
#define yylex           flc_src_lex
#define yyerror         flc_src_error
#define yylval          flc_src_lval
#define yychar          flc_src_char
#define yydebug         flc_src_debug
#define yynerrs         flc_src_nerrs


/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Line 189 of yacc.c  */
#line 124 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     UNB = 264,
     END = 265,
     DEF = 266,
     WHILE = 267,
     BREAK = 268,
     CONTINUE = 269,
     DROPPING = 270,
     IF = 271,
     ELSE = 272,
     ELIF = 273,
     FOR = 274,
     FORFIRST = 275,
     FORLAST = 276,
     FORMIDDLE = 277,
     SWITCH = 278,
     CASE = 279,
     DEFAULT = 280,
     SELECT = 281,
     SELF = 282,
     FSELF = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     INIT = 292,
     LOAD = 293,
     LAUNCH = 294,
     CONST_KW = 295,
     EXPORT = 296,
     IMPORT = 297,
     DIRECTIVE = 298,
     COLON = 299,
     FUNCDECL = 300,
     STATIC = 301,
     INNERFUNC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     ENUM = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_AS = 331,
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     OP_EXEQ = 344,
     PROVIDES = 345,
     OP_NOTIN = 346,
     OP_IN = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     CAP_XOROOB = 361,
     CAP_ISOOB = 362,
     CAP_DEOOB = 363,
     CAP_OOB = 364,
     CAP_EVAL = 365,
     TILDE = 366,
     NEG = 367,
     AMPER = 368,
     DECREMENT = 369,
     INCREMENT = 370,
     DOLLAR = 371
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define UNB 264
#define END 265
#define DEF 266
#define WHILE 267
#define BREAK 268
#define CONTINUE 269
#define DROPPING 270
#define IF 271
#define ELSE 272
#define ELIF 273
#define FOR 274
#define FORFIRST 275
#define FORLAST 276
#define FORMIDDLE 277
#define SWITCH 278
#define CASE 279
#define DEFAULT 280
#define SELECT 281
#define SELF 282
#define FSELF 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define INIT 292
#define LOAD 293
#define LAUNCH 294
#define CONST_KW 295
#define EXPORT 296
#define IMPORT 297
#define DIRECTIVE 298
#define COLON 299
#define FUNCDECL 300
#define STATIC 301
#define INNERFUNC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define ENUM 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_AS 331
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define OP_EXEQ 344
#define PROVIDES 345
#define OP_NOTIN 346
#define OP_IN 347
#define DIESIS 348
#define ATSIGN 349
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define CAP_XOROOB 361
#define CAP_ISOOB 362
#define CAP_DEOOB 363
#define CAP_OOB 364
#define CAP_EVAL 365
#define TILDE 366
#define NEG 367
#define AMPER 368
#define DECREMENT 369
#define INCREMENT 370
#define DOLLAR 371




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union 
/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t
{

/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"

   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
   Falcon::List *fal_genericList;



/* Line 214 of yacc.c  */
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 424 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6811

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  117
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  167
/* YYNRULES -- Number of rules.  */
#define YYNRULES  470
/* YYNRULES -- Number of states.  */
#define YYNSTATES  852

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   371

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   133,   137,   141,   145,   146,   152,   155,
     159,   163,   167,   171,   172,   180,   184,   188,   189,   191,
     192,   199,   202,   206,   210,   214,   218,   219,   221,   222,
     226,   229,   233,   234,   239,   243,   247,   248,   251,   254,
     258,   261,   265,   269,   274,   279,   285,   289,   290,   295,
     296,   303,   308,   313,   317,   318,   321,   324,   325,   328,
     330,   332,   334,   336,   340,   344,   348,   351,   355,   358,
     362,   366,   368,   369,   376,   380,   384,   385,   392,   396,
     400,   401,   408,   412,   416,   417,   424,   428,   432,   433,
     436,   440,   442,   443,   449,   450,   456,   457,   463,   464,
     470,   471,   472,   476,   477,   479,   482,   485,   488,   490,
     494,   496,   498,   500,   504,   506,   507,   514,   518,   522,
     523,   526,   530,   532,   533,   539,   540,   546,   547,   553,
     554,   560,   562,   566,   567,   569,   571,   575,   576,   583,
     586,   590,   591,   593,   595,   598,   601,   604,   609,   613,
     619,   623,   625,   629,   631,   633,   637,   641,   647,   650,
     656,   657,   665,   669,   675,   676,   683,   686,   687,   689,
     693,   695,   696,   697,   703,   704,   708,   711,   715,   718,
     722,   726,   730,   736,   742,   746,   749,   753,   757,   759,
     763,   767,   773,   779,   787,   795,   803,   811,   816,   821,
     826,   831,   838,   845,   849,   854,   859,   861,   865,   869,
     873,   875,   879,   883,   887,   891,   892,   900,   904,   907,
     908,   912,   913,   919,   920,   923,   925,   929,   932,   933,
     936,   940,   941,   944,   946,   948,   950,   952,   954,   956,
     957,   965,   971,   976,   981,   986,   991,   992,   995,   997,
     999,  1000,  1008,  1009,  1012,  1014,  1019,  1021,  1024,  1026,
    1028,  1029,  1037,  1040,  1043,  1044,  1047,  1049,  1051,  1053,
    1055,  1057,  1058,  1063,  1065,  1067,  1070,  1074,  1078,  1080,
    1083,  1087,  1091,  1093,  1095,  1097,  1099,  1101,  1103,  1105,
    1107,  1109,  1111,  1113,  1115,  1117,  1119,  1121,  1123,  1125,
    1127,  1128,  1130,  1132,  1134,  1137,  1140,  1143,  1147,  1151,
    1155,  1158,  1162,  1167,  1172,  1177,  1182,  1187,  1192,  1197,
    1202,  1207,  1212,  1217,  1220,  1224,  1227,  1230,  1233,  1236,
    1240,  1244,  1248,  1252,  1256,  1260,  1264,  1268,  1271,  1275,
    1279,  1283,  1286,  1289,  1292,  1295,  1298,  1301,  1304,  1307,
    1310,  1312,  1314,  1316,  1318,  1320,  1322,  1325,  1327,  1332,
    1338,  1342,  1344,  1346,  1350,  1356,  1360,  1364,  1368,  1372,
    1376,  1380,  1384,  1388,  1392,  1396,  1400,  1404,  1408,  1413,
    1418,  1424,  1432,  1437,  1441,  1442,  1449,  1450,  1457,  1458,
    1465,  1470,  1474,  1477,  1480,  1483,  1486,  1487,  1494,  1500,
    1506,  1511,  1515,  1518,  1522,  1526,  1529,  1533,  1537,  1541,
    1545,  1550,  1552,  1556,  1558,  1562,  1563,  1565,  1567,  1571,
    1575
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     118,     0,    -1,   119,    -1,    -1,   119,   120,    -1,   121,
      -1,    10,     3,    -1,    24,     1,     3,    -1,   123,    -1,
     222,    -1,   202,    -1,   225,    -1,   248,    -1,   243,    -1,
     124,    -1,   216,    -1,   217,    -1,   219,    -1,     4,    -1,
      98,     4,    -1,    38,     6,     3,    -1,    38,     7,     3,
      -1,    38,     1,     3,    -1,   125,    -1,   220,    -1,     3,
      -1,    45,     1,     3,    -1,    34,     1,     3,    -1,    32,
       1,     3,    -1,     1,     3,    -1,   263,     3,    -1,   279,
      75,   263,     3,    -1,   279,    75,   263,    78,   279,     3,
      -1,   128,    -1,   129,    -1,   133,    -1,   150,    -1,   166,
      -1,   181,    -1,   136,    -1,   147,    -1,   148,    -1,   192,
      -1,   201,    -1,   257,    -1,   253,    -1,   215,    -1,   157,
      -1,   158,    -1,   159,    -1,   127,    -1,   126,    78,   127,
      -1,   260,    -1,   260,    75,   263,    -1,    11,   126,     3,
      -1,    11,     1,     3,    -1,    -1,   131,   130,   146,    10,
       3,    -1,   132,   124,    -1,    12,   263,     3,    -1,    12,
       1,     3,    -1,    12,   263,    44,    -1,    12,     1,    44,
      -1,    -1,    50,     3,   134,   146,    10,   135,     3,    -1,
      50,    44,   124,    -1,    50,     1,     3,    -1,    -1,   263,
      -1,    -1,   138,   137,   146,   140,    10,     3,    -1,   139,
     124,    -1,    16,   263,     3,    -1,    16,     1,     3,    -1,
      16,   263,    44,    -1,    16,     1,    44,    -1,    -1,   143,
      -1,    -1,   142,   141,   146,    -1,    17,     3,    -1,    17,
       1,     3,    -1,    -1,   145,   144,   146,   140,    -1,    18,
     263,     3,    -1,    18,     1,     3,    -1,    -1,   146,   124,
      -1,    13,     3,    -1,    13,     1,     3,    -1,    14,     3,
      -1,    14,    15,     3,    -1,    14,     1,     3,    -1,    19,
     282,    92,   263,    -1,    19,   260,    75,   153,    -1,    19,
     282,    92,     1,     3,    -1,    19,     1,     3,    -1,    -1,
     149,    44,   151,   124,    -1,    -1,   149,     3,   152,   155,
      10,     3,    -1,   263,    77,   263,   154,    -1,   263,    77,
     263,     1,    -1,   263,    77,     1,    -1,    -1,    78,   263,
      -1,    78,     1,    -1,    -1,   156,   155,    -1,   124,    -1,
     160,    -1,   162,    -1,   164,    -1,    48,   263,     3,    -1,
      48,     1,     3,    -1,   104,   279,     3,    -1,   104,     3,
      -1,    86,   279,     3,    -1,    86,     3,    -1,   104,     1,
       3,    -1,    86,     1,     3,    -1,    54,    -1,    -1,    20,
       3,   161,   146,    10,     3,    -1,    20,    44,   124,    -1,
      20,     1,     3,    -1,    -1,    21,     3,   163,   146,    10,
       3,    -1,    21,    44,   124,    -1,    21,     1,     3,    -1,
      -1,    22,     3,   165,   146,    10,     3,    -1,    22,    44,
     124,    -1,    22,     1,     3,    -1,    -1,   168,   167,   169,
     175,    10,     3,    -1,    23,   263,     3,    -1,    23,     1,
       3,    -1,    -1,   169,   170,    -1,   169,     1,     3,    -1,
       3,    -1,    -1,    24,   179,     3,   171,   146,    -1,    -1,
      24,   179,    44,   172,   124,    -1,    -1,    24,     1,     3,
     173,   146,    -1,    -1,    24,     1,    44,   174,   124,    -1,
      -1,    -1,   177,   176,   178,    -1,    -1,    25,    -1,    25,
       1,    -1,     3,   146,    -1,    44,   124,    -1,   180,    -1,
     179,    78,   180,    -1,     8,    -1,   122,    -1,     7,    -1,
     122,    77,   122,    -1,     6,    -1,    -1,   183,   182,   184,
     175,    10,     3,    -1,    26,   263,     3,    -1,    26,     1,
       3,    -1,    -1,   184,   185,    -1,   184,     1,     3,    -1,
       3,    -1,    -1,    24,   190,     3,   186,   146,    -1,    -1,
      24,   190,    44,   187,   124,    -1,    -1,    24,     1,     3,
     188,   146,    -1,    -1,    24,     1,    44,   189,   124,    -1,
     191,    -1,   190,    78,   191,    -1,    -1,     4,    -1,     6,
      -1,    29,    44,   124,    -1,    -1,   194,   193,   146,   195,
      10,     3,    -1,    29,     3,    -1,    29,     1,     3,    -1,
      -1,   196,    -1,   197,    -1,   196,   197,    -1,   198,   146,
      -1,    30,     3,    -1,    30,    92,   260,     3,    -1,    30,
     199,     3,    -1,    30,   199,    92,   260,     3,    -1,    30,
       1,     3,    -1,   200,    -1,   199,    78,   200,    -1,     4,
      -1,     6,    -1,    31,   263,     3,    -1,    31,     1,     3,
      -1,   203,   210,   146,    10,     3,    -1,   205,   124,    -1,
     207,    56,   208,    55,     3,    -1,    -1,   207,    56,   208,
       1,   204,    55,     3,    -1,   207,     1,     3,    -1,   207,
      56,   208,    55,    44,    -1,    -1,   207,    56,     1,   206,
      55,    44,    -1,    45,     6,    -1,    -1,   209,    -1,   208,
      78,   209,    -1,     6,    -1,    -1,    -1,   213,   211,   146,
      10,     3,    -1,    -1,   214,   212,   124,    -1,    46,     3,
      -1,    46,     1,     3,    -1,    46,    44,    -1,    46,     1,
      44,    -1,    39,   265,     3,    -1,    39,     1,     3,    -1,
      40,     6,    75,   259,     3,    -1,    40,     6,    75,     1,
       3,    -1,    40,     1,     3,    -1,    41,     3,    -1,    41,
     218,     3,    -1,    41,     1,     3,    -1,     6,    -1,   218,
      78,     6,    -1,    42,   221,     3,    -1,    42,   221,    33,
       6,     3,    -1,    42,   221,    33,     7,     3,    -1,    42,
     221,    33,     6,    76,     6,     3,    -1,    42,   221,    33,
       7,    76,     6,     3,    -1,    42,   221,    33,     6,    92,
       6,     3,    -1,    42,   221,    33,     7,    92,     6,     3,
      -1,    42,     6,     1,     3,    -1,    42,   221,     1,     3,
      -1,    42,    33,     6,     3,    -1,    42,    33,     7,     3,
      -1,    42,    33,     6,    92,     6,     3,    -1,    42,    33,
       7,    92,     6,     3,    -1,    42,     1,     3,    -1,     6,
      44,   259,     3,    -1,     6,    44,     1,     3,    -1,     6,
      -1,   221,    78,     6,    -1,    43,   223,     3,    -1,    43,
       1,     3,    -1,   224,    -1,   223,    78,   224,    -1,     6,
      75,     6,    -1,     6,    75,     7,    -1,     6,    75,   122,
      -1,    -1,    32,     6,   226,   227,   234,    10,     3,    -1,
     228,   230,     3,    -1,     1,     3,    -1,    -1,    56,   208,
      55,    -1,    -1,    56,   208,     1,   229,    55,    -1,    -1,
      33,   231,    -1,   232,    -1,   231,    78,   232,    -1,     6,
     233,    -1,    -1,    56,    55,    -1,    56,   279,    55,    -1,
      -1,   234,   235,    -1,     3,    -1,   202,    -1,   238,    -1,
     239,    -1,   236,    -1,   220,    -1,    -1,    37,     3,   237,
     210,   146,    10,     3,    -1,    46,     6,    75,   259,     3,
      -1,     6,    75,   263,     3,    -1,   240,   241,    10,     3,
      -1,    58,     6,    57,     3,    -1,    58,    37,    57,     3,
      -1,    -1,   241,   242,    -1,     3,    -1,   202,    -1,    -1,
      51,     6,   244,     3,   245,    10,     3,    -1,    -1,   245,
     246,    -1,     3,    -1,     6,    75,   259,   247,    -1,   220,
      -1,     6,   247,    -1,     3,    -1,    78,    -1,    -1,    34,
       6,   249,   250,   251,    10,     3,    -1,   230,     3,    -1,
       1,     3,    -1,    -1,   251,   252,    -1,     3,    -1,   202,
      -1,   238,    -1,   236,    -1,   220,    -1,    -1,    36,   254,
     255,     3,    -1,   256,    -1,     1,    -1,   256,     1,    -1,
     255,    78,   256,    -1,   255,    78,     1,    -1,     6,    -1,
      35,     3,    -1,    35,   263,     3,    -1,    35,     1,     3,
      -1,     8,    -1,     9,    -1,    52,    -1,    53,    -1,     4,
      -1,     5,    -1,     7,    -1,     8,    -1,     9,    -1,    52,
      -1,    53,    -1,   122,    -1,     5,    -1,     7,    -1,     6,
      -1,   260,    -1,    27,    -1,    28,    -1,    -1,     3,    -1,
     258,    -1,   261,    -1,   113,     6,    -1,   113,     4,    -1,
     113,    27,    -1,   113,    59,     6,    -1,   113,    59,     4,
      -1,   113,    59,    27,    -1,    98,   263,    -1,     6,    63,
     263,    -1,   263,    99,   262,   263,    -1,   263,    98,   262,
     263,    -1,   263,   102,   262,   263,    -1,   263,   101,   262,
     263,    -1,   263,   100,   262,   263,    -1,   263,   103,   262,
     263,    -1,   263,    97,   262,   263,    -1,   263,    96,   262,
     263,    -1,   263,    95,   262,   263,    -1,   263,   105,   262,
     263,    -1,   263,   104,   262,   263,    -1,   111,   263,    -1,
     263,    87,   263,    -1,   263,   115,    -1,   115,   263,    -1,
     263,   114,    -1,   114,   263,    -1,   263,    88,   263,    -1,
     263,    89,   263,    -1,   263,    86,   263,    -1,   263,    85,
     263,    -1,   263,    84,   263,    -1,   263,    83,   263,    -1,
     263,    81,   263,    -1,   263,    80,   263,    -1,    82,   263,
      -1,   263,    92,   263,    -1,   263,    91,   263,    -1,   263,
      90,     6,    -1,   116,   260,    -1,   116,     4,    -1,    94,
     263,    -1,    93,   263,    -1,   110,   263,    -1,   109,   263,
      -1,   108,   263,    -1,   107,   263,    -1,   106,   263,    -1,
     267,    -1,   269,    -1,   273,    -1,   265,    -1,   275,    -1,
     277,    -1,   263,   264,    -1,   276,    -1,   263,    58,   263,
      57,    -1,   263,    58,   102,   263,    57,    -1,   263,    59,
       6,    -1,   278,    -1,   264,    -1,   263,    75,   263,    -1,
     263,    75,   263,    78,   279,    -1,   263,    74,   263,    -1,
     263,    73,   263,    -1,   263,    72,   263,    -1,   263,    71,
     263,    -1,   263,    70,   263,    -1,   263,    64,   263,    -1,
     263,    69,   263,    -1,   263,    68,   263,    -1,   263,    67,
     263,    -1,   263,    65,   263,    -1,   263,    66,   263,    -1,
      56,   263,    55,    -1,    58,    44,    57,    -1,    58,   263,
      44,    57,    -1,    58,    44,   263,    57,    -1,    58,   263,
      44,   263,    57,    -1,    58,   263,    44,   263,    44,   263,
      57,    -1,   263,    56,   279,    55,    -1,   263,    56,    55,
      -1,    -1,   263,    56,   279,     1,   266,    55,    -1,    -1,
      45,   268,   271,   210,   146,    10,    -1,    -1,    60,   270,
     272,   210,   146,    61,    -1,    56,   208,    55,     3,    -1,
      56,   208,     1,    -1,     1,     3,    -1,   208,    62,    -1,
     208,     1,    -1,     1,    62,    -1,    -1,    47,   274,   271,
     210,   146,    10,    -1,   263,    79,   263,    44,   263,    -1,
     263,    79,   263,    44,     1,    -1,   263,    79,   263,     1,
      -1,   263,    79,     1,    -1,    58,    57,    -1,    58,   279,
      57,    -1,    58,   279,     1,    -1,    49,    57,    -1,    49,
     280,    57,    -1,    49,   280,     1,    -1,    58,    62,    57,
      -1,    58,   283,    57,    -1,    58,   283,     1,    57,    -1,
     263,    -1,   279,    78,   263,    -1,   263,    -1,   280,   281,
     263,    -1,    -1,    78,    -1,   260,    -1,   282,    78,   260,
      -1,   263,    62,   263,    -1,   283,    78,   263,    62,   263,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   200,   202,   206,   207,   208,   212,   213,
     214,   219,   224,   229,   234,   239,   240,   241,   245,   246,
     250,   256,   262,   269,   270,   271,   272,   273,   274,   275,
     280,   291,   308,   322,   323,   324,   325,   326,   327,   328,
     329,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     342,   343,   347,   355,   365,   367,   372,   372,   386,   394,
     395,   399,   400,   404,   404,   419,   425,   432,   433,   437,
     437,   452,   462,   463,   467,   468,   472,   474,   475,   475,
     484,   485,   490,   490,   502,   503,   506,   508,   514,   523,
     531,   541,   550,   558,   563,   571,   576,   586,   585,   606,
     605,   627,   633,   637,   644,   645,   646,   649,   651,   655,
     662,   663,   664,   668,   681,   689,   693,   699,   705,   712,
     717,   726,   736,   736,   750,   759,   763,   763,   776,   785,
     789,   789,   805,   814,   818,   818,   835,   836,   843,   845,
     846,   850,   852,   851,   862,   862,   874,   874,   886,   886,
     902,   905,   904,   917,   918,   919,   922,   923,   929,   930,
     934,   943,   955,   966,   977,   998,   998,  1015,  1016,  1023,
    1025,  1026,  1030,  1032,  1031,  1042,  1042,  1055,  1055,  1067,
    1067,  1085,  1086,  1089,  1090,  1102,  1123,  1130,  1129,  1148,
    1149,  1152,  1154,  1158,  1159,  1163,  1168,  1186,  1206,  1216,
    1227,  1235,  1236,  1240,  1252,  1275,  1276,  1283,  1293,  1302,
    1303,  1303,  1307,  1311,  1312,  1312,  1319,  1441,  1443,  1444,
    1448,  1463,  1466,  1465,  1477,  1476,  1491,  1492,  1496,  1497,
    1506,  1510,  1518,  1528,  1533,  1545,  1554,  1561,  1569,  1574,
    1582,  1587,  1592,  1597,  1617,  1636,  1641,  1646,  1651,  1665,
    1670,  1675,  1680,  1685,  1694,  1699,  1706,  1712,  1724,  1729,
    1737,  1738,  1742,  1746,  1750,  1764,  1763,  1826,  1829,  1835,
    1837,  1838,  1838,  1844,  1846,  1850,  1851,  1855,  1879,  1880,
    1881,  1888,  1890,  1894,  1895,  1898,  1917,  1937,  1938,  1942,
    1942,  1976,  1998,  2026,  2038,  2046,  2059,  2061,  2066,  2067,
    2079,  2078,  2122,  2124,  2128,  2129,  2133,  2134,  2141,  2141,
    2150,  2149,  2216,  2217,  2223,  2225,  2229,  2230,  2233,  2252,
    2253,  2262,  2261,  2279,  2280,  2285,  2290,  2291,  2298,  2314,
    2315,  2316,  2324,  2325,  2326,  2327,  2328,  2329,  2330,  2334,
    2335,  2336,  2337,  2338,  2339,  2340,  2344,  2362,  2363,  2364,
    2384,  2386,  2390,  2391,  2392,  2393,  2394,  2395,  2396,  2397,
    2398,  2399,  2400,  2426,  2427,  2447,  2471,  2488,  2489,  2490,
    2491,  2492,  2493,  2494,  2495,  2496,  2497,  2498,  2499,  2500,
    2501,  2502,  2503,  2504,  2505,  2506,  2507,  2508,  2509,  2510,
    2511,  2512,  2513,  2514,  2515,  2516,  2517,  2518,  2519,  2520,
    2521,  2522,  2523,  2524,  2525,  2526,  2528,  2533,  2537,  2542,
    2548,  2557,  2558,  2560,  2565,  2572,  2573,  2574,  2575,  2576,
    2577,  2578,  2579,  2580,  2581,  2582,  2583,  2588,  2591,  2594,
    2597,  2600,  2606,  2612,  2617,  2617,  2627,  2626,  2669,  2668,
    2720,  2721,  2725,  2732,  2733,  2737,  2745,  2744,  2793,  2798,
    2805,  2812,  2822,  2823,  2827,  2835,  2836,  2840,  2849,  2850,
    2851,  2859,  2860,  2864,  2865,  2868,  2869,  2872,  2878,  2885,
    2886
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "UNB", "END", "DEF", "WHILE", "BREAK", "CONTINUE",
  "DROPPING", "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST",
  "FORMIDDLE", "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF",
  "TRY", "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "INIT", "LOAD", "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE",
  "COLON", "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP",
  "ENUM", "TRUE_TOKEN", "FALSE_TOKEN", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH",
  "CLOSE_GRAPH", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "OP_EXEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN",
  "CAP_CAP", "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT",
  "SLASH", "STAR", "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB",
  "CAP_DEOOB", "CAP_OOB", "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT",
  "INCREMENT", "DOLLAR", "$accept", "input", "body", "line",
  "toplevel_statement", "INTNUM_WITH_MINUS", "load_statement", "statement",
  "base_statement", "assignment_def_list", "assignment_def_list_element",
  "def_statement", "while_statement", "$@1", "while_decl",
  "while_short_decl", "loop_statement", "$@2", "loop_terminator",
  "if_statement", "$@3", "if_decl", "if_short_decl", "elif_or_else", "$@4",
  "else_decl", "elif_statement", "$@5", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "forin_header",
  "forin_statement", "$@6", "$@7", "for_to_expr", "for_to_step_clause",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "$@8", "last_loop_block", "$@9", "middle_loop_block", "$@10",
  "switch_statement", "$@11", "switch_decl", "case_list", "case_statement",
  "$@12", "$@13", "$@14", "$@15", "default_statement", "$@16",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "$@17", "select_decl", "selcase_list",
  "selcase_statement", "$@18", "$@19", "$@20", "$@21",
  "selcase_expression_list", "selcase_element", "try_statement", "$@22",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "$@23",
  "func_decl_short", "$@24", "func_begin", "param_list", "param_symbol",
  "static_block", "$@25", "$@26", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "$@27", "class_def_inner",
  "class_param_list", "$@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "$@29", "property_decl", "state_decl",
  "state_heading", "state_statement_list", "state_statement",
  "enum_statement", "$@30", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "$@31", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "$@32",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom_non_minus", "const_atom", "atomic_symbol", "var_atom",
  "OPT_EOL", "expression", "range_decl", "func_call", "$@33",
  "nameless_func", "$@34", "nameless_block", "$@35",
  "nameless_func_decl_inner", "nameless_block_decl_inner", "innerfunc",
  "$@36", "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "listpar_comma",
  "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   117,   118,   119,   119,   120,   120,   120,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   122,   122,
     123,   123,   123,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     126,   126,   127,   127,   128,   128,   130,   129,   129,   131,
     131,   132,   132,   134,   133,   133,   133,   135,   135,   137,
     136,   136,   138,   138,   139,   139,   140,   140,   141,   140,
     142,   142,   144,   143,   145,   145,   146,   146,   147,   147,
     148,   148,   148,   149,   149,   149,   149,   151,   150,   152,
     150,   153,   153,   153,   154,   154,   154,   155,   155,   156,
     156,   156,   156,   157,   157,   158,   158,   158,   158,   158,
     158,   159,   161,   160,   160,   160,   163,   162,   162,   162,
     165,   164,   164,   164,   167,   166,   168,   168,   169,   169,
     169,   170,   171,   170,   172,   170,   173,   170,   174,   170,
     175,   176,   175,   177,   177,   177,   178,   178,   179,   179,
     180,   180,   180,   180,   180,   182,   181,   183,   183,   184,
     184,   184,   185,   186,   185,   187,   185,   188,   185,   189,
     185,   190,   190,   191,   191,   191,   192,   193,   192,   194,
     194,   195,   195,   196,   196,   197,   198,   198,   198,   198,
     198,   199,   199,   200,   200,   201,   201,   202,   202,   203,
     204,   203,   203,   205,   206,   205,   207,   208,   208,   208,
     209,   210,   211,   210,   212,   210,   213,   213,   214,   214,
     215,   215,   216,   216,   216,   217,   217,   217,   218,   218,
     219,   219,   219,   219,   219,   219,   219,   219,   219,   219,
     219,   219,   219,   219,   220,   220,   221,   221,   222,   222,
     223,   223,   224,   224,   224,   226,   225,   227,   227,   228,
     228,   229,   228,   230,   230,   231,   231,   232,   233,   233,
     233,   234,   234,   235,   235,   235,   235,   235,   235,   237,
     236,   238,   238,   239,   240,   240,   241,   241,   242,   242,
     244,   243,   245,   245,   246,   246,   246,   246,   247,   247,
     249,   248,   250,   250,   251,   251,   252,   252,   252,   252,
     252,   254,   253,   255,   255,   255,   255,   255,   256,   257,
     257,   257,   258,   258,   258,   258,   258,   258,   258,   259,
     259,   259,   259,   259,   259,   259,   260,   261,   261,   261,
     262,   262,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   264,   264,   264,
     264,   264,   265,   265,   266,   265,   268,   267,   270,   269,
     271,   271,   271,   272,   272,   272,   274,   273,   275,   275,
     275,   275,   276,   276,   276,   277,   277,   277,   278,   278,
     278,   279,   279,   280,   280,   281,   281,   282,   282,   283,
     283
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     3,     3,     3,     0,     5,     2,     3,
       3,     3,     3,     0,     7,     3,     3,     0,     1,     0,
       6,     2,     3,     3,     3,     3,     0,     1,     0,     3,
       2,     3,     0,     4,     3,     3,     0,     2,     2,     3,
       2,     3,     3,     4,     4,     5,     3,     0,     4,     0,
       6,     4,     4,     3,     0,     2,     2,     0,     2,     1,
       1,     1,     1,     3,     3,     3,     2,     3,     2,     3,
       3,     1,     0,     6,     3,     3,     0,     6,     3,     3,
       0,     6,     3,     3,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       0,     0,     3,     0,     1,     2,     2,     2,     1,     3,
       1,     1,     1,     3,     1,     0,     6,     3,     3,     0,
       2,     3,     1,     0,     5,     0,     5,     0,     5,     0,
       5,     1,     3,     0,     1,     1,     3,     0,     6,     2,
       3,     0,     1,     1,     2,     2,     2,     4,     3,     5,
       3,     1,     3,     1,     1,     3,     3,     5,     2,     5,
       0,     7,     3,     5,     0,     6,     2,     0,     1,     3,
       1,     0,     0,     5,     0,     3,     2,     3,     2,     3,
       3,     3,     5,     5,     3,     2,     3,     3,     1,     3,
       3,     5,     5,     7,     7,     7,     7,     4,     4,     4,
       4,     6,     6,     3,     4,     4,     1,     3,     3,     3,
       1,     3,     3,     3,     3,     0,     7,     3,     2,     0,
       3,     0,     5,     0,     2,     1,     3,     2,     0,     2,
       3,     0,     2,     1,     1,     1,     1,     1,     1,     0,
       7,     5,     4,     4,     4,     4,     0,     2,     1,     1,
       0,     7,     0,     2,     1,     4,     1,     2,     1,     1,
       0,     7,     2,     2,     0,     2,     1,     1,     1,     1,
       1,     0,     4,     1,     1,     2,     3,     3,     1,     2,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       0,     1,     1,     1,     2,     2,     2,     3,     3,     3,
       2,     3,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     2,     3,     2,     2,     2,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     2,     3,     3,
       3,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     2,     1,     4,     5,
       3,     1,     1,     3,     5,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     4,     4,
       5,     7,     4,     3,     0,     6,     0,     6,     0,     6,
       4,     3,     2,     2,     2,     2,     0,     6,     5,     5,
       4,     3,     2,     3,     3,     2,     3,     3,     3,     3,
       4,     1,     3,     1,     3,     0,     1,     1,     3,     3,
       5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   336,   337,   346,   338,
     332,   333,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   348,   349,     0,     0,     0,     0,     0,   321,
       0,     0,     0,     0,     0,     0,     0,   446,     0,     0,
       0,     0,   334,   335,   121,     0,     0,   438,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    14,    23,    33,
      34,    56,     0,    35,    39,    69,     0,    40,    41,     0,
      36,    47,    48,    49,    37,   134,    38,   165,    42,   187,
      43,    10,   221,     0,     0,    46,    15,    16,    17,    24,
       9,    11,    13,    12,    45,    44,   352,   347,   353,   461,
     412,   403,   400,   401,   402,   404,   407,   405,   411,     0,
      29,     0,     0,     6,     0,   346,     0,    50,    52,     0,
     346,   436,     0,     0,    88,     0,    90,     0,     0,     0,
       0,   467,     0,     0,     0,     0,     0,     0,     0,   189,
       0,     0,     0,     0,   265,     0,   310,     0,   329,     0,
       0,     0,     0,     0,     0,     0,   403,     0,     0,     0,
     235,   238,     0,     0,     0,     0,     0,     0,     0,     0,
     260,     0,   216,     0,     0,     0,     0,   455,   463,     0,
       0,    63,     0,   300,     0,     0,   452,     0,   461,     0,
       0,     0,   387,     0,   118,   461,     0,   394,   393,   360,
       0,   116,     0,   399,   398,   397,   396,   395,   373,   355,
     354,   356,     0,   378,   376,   392,   391,    86,     0,     0,
       0,    58,    86,    71,    99,    97,   138,   169,    86,     0,
      86,   222,   224,   208,     0,     0,    30,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   350,   350,   350,   350,   350,
     350,   350,   350,   350,   350,   350,   377,   375,   406,     0,
       0,     0,    18,   344,   345,   339,   340,   341,   342,     0,
     343,     0,   361,    55,    54,     0,     0,    60,    62,    59,
      61,    89,    92,    91,    73,    75,    72,    74,    96,     0,
       0,     0,   137,   136,     7,   168,   167,   190,   186,   206,
     205,    28,     0,    27,     0,   331,   330,   324,   328,     0,
       0,    22,    20,    21,   231,   230,   234,     0,   237,   236,
       0,   253,     0,     0,     0,     0,   240,     0,     0,   259,
       0,   258,     0,    26,     0,   217,   221,   221,   114,   113,
     457,   456,   466,     0,    66,    86,    65,     0,   426,   427,
       0,   458,     0,     0,   454,   453,     0,   459,     0,     0,
     220,     0,   218,   221,   120,   117,   119,   115,   358,   357,
     359,     0,     0,     0,     0,     0,     0,     0,     0,   226,
     228,     0,    86,     0,   212,   214,     0,   433,     0,     0,
       0,   410,   420,   424,   425,   423,   422,   421,   419,   418,
     417,   416,   415,   413,   451,     0,   386,   385,   384,   383,
     382,   381,   374,   379,   380,   390,   389,   388,   351,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   462,   255,    19,   254,    51,    53,    94,     0,   468,
       0,    93,     0,   217,   281,   273,     0,     0,     0,   314,
     322,     0,   325,     0,     0,   239,   247,   249,     0,   250,
       0,   248,     0,     0,   257,   262,   263,   264,   261,   442,
       0,    86,    86,   464,     0,   302,   429,   428,     0,   469,
     460,     0,   445,   444,   443,     0,    86,     0,    87,     0,
       0,     0,    78,    77,    82,     0,     0,     0,   109,     0,
       0,   110,   111,   112,    98,     0,   141,     0,     0,   139,
       0,   151,     0,   172,     0,     0,   170,     0,     0,   192,
     193,    86,   227,   229,     0,     0,   225,     0,   210,     0,
     434,   432,     0,   408,     0,   450,     0,   370,   369,   368,
     363,   362,   366,   365,   364,   367,   372,   371,    31,     0,
       0,    95,   268,     0,     0,     0,   313,   278,   274,   275,
     312,     0,   327,   326,   233,   232,     0,     0,   241,     0,
       0,   242,     0,     0,   441,     0,     0,     0,    67,     0,
       0,   430,     0,   219,     0,    57,     0,    80,     0,     0,
       0,    86,    86,     0,   122,     0,     0,   126,     0,     0,
     130,     0,     0,   108,   140,     0,   164,   162,   160,   161,
       0,   158,   155,     0,     0,   171,     0,   184,   185,     0,
     181,     0,     0,   196,   203,   204,     0,     0,   201,     0,
     194,     0,   207,     0,     0,     0,   209,   213,     0,   409,
     414,   449,   448,     0,   103,     0,   271,   270,   283,     0,
       0,     0,     0,     0,     0,   284,   288,   282,   287,   285,
     286,   296,   267,     0,   277,     0,   316,     0,   317,   320,
     319,   318,   315,   251,   252,     0,     0,     0,     0,   440,
     437,   447,     0,    68,   304,     0,     0,   306,   303,     0,
     470,   439,    81,    85,    84,    70,     0,     0,   125,    86,
     124,   129,    86,   128,   133,    86,   132,   100,   146,   148,
       0,   142,   144,     0,   135,    86,     0,   152,   177,   179,
     173,   175,   183,   166,   200,     0,   198,     0,     0,   188,
     223,   215,     0,   435,    32,   102,     0,   101,     0,     0,
     266,   289,     0,     0,     0,     0,   279,     0,   276,   311,
     243,   245,   244,   246,    64,   308,     0,   309,   307,   301,
     431,    83,     0,     0,     0,    86,     0,   163,    86,     0,
     159,     0,   157,    86,     0,    86,     0,   182,   197,   202,
       0,   211,   106,   105,   272,     0,   221,     0,     0,     0,
     298,     0,   299,   297,   280,     0,     0,     0,     0,     0,
     149,     0,   145,     0,   180,     0,   176,   199,   292,    86,
       0,   294,   295,   293,   305,   123,   127,   131,     0,   291,
       0,   290
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    64,    65,   300,    66,   518,    68,   126,
     127,    69,    70,   227,    71,    72,    73,   375,   712,    74,
     232,    75,    76,   521,   621,   522,   523,   622,   524,   401,
      77,    78,    79,    80,   404,   403,   467,   767,   529,   530,
      81,    82,    83,   531,   729,   532,   732,   533,   735,    84,
     236,    85,   405,   539,   798,   799,   795,   796,   540,   644,
     541,   747,   640,   641,    86,   237,    87,   406,   546,   805,
     806,   803,   804,   649,   650,    88,   238,    89,   548,   549,
     550,   551,   657,   658,    90,    91,    92,   665,    93,   557,
      94,   391,   392,   240,   412,   413,   241,   242,    95,    96,
      97,   172,    98,    99,   176,   100,   179,   180,   101,   332,
     474,   475,   768,   478,   588,   589,   694,   584,   687,   688,
     816,   689,   690,   691,   775,   823,   102,   377,   609,   718,
     788,   103,   334,   479,   591,   702,   104,   160,   339,   340,
     105,   106,   301,   107,   108,   449,   109,   110,   111,   668,
     112,   183,   113,   201,   366,   393,   114,   184,   115,   116,
     117,   118,   119,   189,   373,   142,   200
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -575
static const yytype_int16 yypact[] =
{
    -575,    77,   869,  -575,    93,  -575,  -575,  -575,    24,  -575,
    -575,  -575,   118,    23,  3715,    94,   280,  3787,   296,  3859,
      87,  3931,  -575,  -575,   222,  4003,   370,   373,  3499,  -575,
     342,  4075,   511,   362,   372,   526,   283,  -575,  4147,  5585,
     270,   180,  -575,  -575,  -575,  5945,  5418,  -575,  5945,  3571,
    5945,  5945,  5945,  3643,  5945,  5945,  5945,  5945,  5945,  5945,
     360,  5945,  5945,   264,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  3427,  -575,  -575,  -575,  3427,  -575,  -575,   323,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,   156,  3427,    26,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  4952,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   276,
    -575,    46,  5945,  -575,   210,  -575,   139,  -575,   147,   385,
     168,  -575,  4775,   271,  -575,   301,  -575,   337,   407,  4848,
     413,   356,   -17,   446,  5004,   475,   486,  5056,   502,  -575,
    3427,   503,  5108,   520,  -575,   523,  -575,   540,  -575,  5160,
     532,   547,   557,   560,   561,  6496,   562,   565,   460,   566,
    -575,  -575,   151,   568,   101,   453,   158,   570,   507,   186,
    -575,   575,  -575,    27,    27,   580,  5212,  -575,  6496,    58,
     593,  -575,  3427,  -575,  6182,  5657,  -575,   541,  6006,   142,
     149,   100,  6696,   597,  -575,  6496,   194,   294,   294,   303,
     598,  -575,   197,   303,   303,   303,   303,   303,   303,  -575,
    -575,  -575,   188,   303,   303,  -575,  -575,  -575,   601,   605,
     125,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   341,
    -575,  -575,  -575,  -575,   606,   140,  -575,  5729,  5513,   604,
    5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,
    5945,  5945,  4219,  5945,  5945,  5945,  5945,  5945,  5945,  5945,
    5945,  5945,   607,  5945,  5945,   609,   609,   609,   609,   609,
     609,   609,   609,   609,   609,   609,  -575,  -575,  -575,  5945,
    5945,   619,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   620,
    -575,   622,  6496,  -575,  -575,   621,  5945,  -575,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  5945,
     621,  4291,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,   521,  -575,   420,  -575,  -575,  -575,  -575,   200,
     184,  -575,  -575,  -575,  -575,  -575,  -575,   123,  -575,  -575,
     627,  -575,   625,    31,    57,   632,  -575,   461,   630,  -575,
      65,  -575,   633,  -575,   634,   636,   156,   156,  -575,  -575,
    -575,  -575,  -575,  5945,  -575,  -575,  -575,   635,  -575,  -575,
    6234,  -575,  5801,  5945,  -575,  -575,   583,  -575,  5945,   581,
    -575,   134,  -575,   156,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  1919,  1571,   991,  3427,   515,   631,  1687,   427,  -575,
    -575,  2035,  -575,  3427,  -575,  -575,   138,  -575,   152,  5945,
    6068,  -575,  6496,  6496,  6496,  6496,  6496,  6496,  6496,  6496,
    6496,  6496,  6496,   232,  -575,  4702,  6646,  6696,   489,   489,
     489,   489,   489,   489,   489,  -575,   294,   294,  -575,  5945,
    5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,  5945,
    4900,  6546,  -575,  -575,  -575,  -575,  6496,  -575,  6286,  -575,
     641,  6496,   643,   636,  -575,   614,   654,   652,   656,  -575,
    -575,   543,  -575,   658,   659,  -575,  -575,  -575,   670,  -575,
     671,  -575,     8,    53,  -575,  -575,  -575,  -575,  -575,  -575,
     204,  -575,  -575,  6496,  2151,  -575,  -575,  -575,  6130,  6496,
    -575,  6340,  -575,  -575,  -575,   636,  -575,   675,  -575,   431,
    4363,   669,  -575,  -575,  -575,   381,   421,   425,  -575,   673,
     991,  -575,  -575,  -575,  -575,   681,  -575,    72,   449,  -575,
     676,  -575,   682,  -575,   154,   677,  -575,   116,   679,   672,
    -575,  -575,  -575,  -575,   710,  2267,  -575,   661,  -575,   444,
    -575,  -575,  6392,  -575,  5945,  -575,  4435,   399,   399,   516,
     549,   549,   405,   405,   405,   299,   303,   303,  -575,  5945,
    4507,  -575,  -575,   208,   484,   714,  -575,   662,   642,  -575,
    -575,   335,  -575,  -575,  -575,  -575,   716,   718,  -575,   717,
     719,  -575,   720,   721,  -575,   727,  2383,  2499,  5945,   531,
    5945,  -575,  5945,  -575,  2615,  -575,   728,  -575,   729,  5264,
     730,  -575,  -575,   732,  -575,  3427,   734,  -575,  3427,   735,
    -575,  3427,   737,  -575,  -575,   471,  -575,  -575,  -575,   645,
     223,  -575,  -575,   738,   492,  -575,   508,  -575,  -575,   225,
    -575,   739,   740,  -575,  -575,  -575,   621,    55,  -575,   742,
    -575,  1803,  -575,   743,   680,   693,  -575,  -575,   694,  -575,
     674,  -575,  6596,   201,  -575,  4640,  -575,  -575,  -575,    37,
     747,   748,   749,   751,   319,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,  5873,  -575,   652,  -575,   755,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,   756,   757,   758,   759,  -575,
    -575,  -575,   760,  6496,  -575,   221,   761,  -575,  -575,  6444,
    6496,  -575,  -575,  -575,  -575,  -575,  2731,  1571,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
      10,  -575,  -575,    15,  -575,  -575,  3427,  -575,  -575,  -575,
    -575,  -575,   551,  -575,  -575,   762,  -575,   552,   621,  -575,
    -575,  -575,   763,  -575,  -575,  -575,  4579,  -575,   699,  5945,
    -575,  -575,   692,   711,   712,   174,  -575,   133,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,    85,  -575,  -575,  -575,
    -575,  -575,  2847,  2963,  3079,  -575,  3427,  -575,  -575,  3427,
    -575,  3195,  -575,  -575,  3427,  -575,  3427,  -575,  -575,  -575,
     767,  -575,  -575,  6496,  -575,  5316,   156,    85,   769,   770,
    -575,   771,  -575,  -575,  -575,   207,   772,   775,   777,  1107,
    -575,  1223,  -575,  1339,  -575,  1455,  -575,  -575,  -575,  -575,
     778,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  3311,  -575,
     779,  -575
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -575,  -575,  -575,  -575,  -575,  -357,  -575,    -2,  -575,  -575,
     478,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,    59,  -575,  -575,  -575,  -575,  -575,    60,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   254,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   379,  -575,
    -575,  -575,  -575,    44,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,  -575,    36,  -575,  -575,  -575,  -575,  -575,
     241,  -575,  -575,    34,  -575,  -574,  -575,  -575,  -575,  -575,
    -575,  -240,   278,  -336,  -575,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,  -575,  -304,  -575,  -575,  -575,   434,  -575,  -575,
    -575,  -575,  -575,   324,  -575,   103,  -575,  -575,  -575,   209,
    -575,   212,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,
     -24,  -575,  -575,  -575,  -575,  -575,  -575,  -575,  -575,   325,
    -575,  -575,  -338,   -11,  -575,   389,   -13,   268,   776,  -575,
    -575,  -575,  -575,  -575,   624,  -575,  -575,  -575,  -575,  -575,
    -575,  -575,   -33,  -575,  -575,  -575,  -575
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -466
static const yytype_int16 yytable[] =
{
      67,   132,   128,   497,   139,   416,   144,   141,   147,   484,
     685,   598,   152,   199,   292,   159,   206,   698,   165,   292,
     212,   636,   637,   638,   124,   186,   188,   244,   364,   125,
     501,   502,   194,   198,   487,   202,   205,   207,   208,   209,
     205,   213,   214,   215,   216,   217,   218,   291,   223,   224,
     292,   293,   226,   294,   295,   296,   601,   516,   756,   370,
     489,   320,  -465,  -465,  -465,  -465,  -465,  -465,   121,   292,
     231,   495,   496,   635,   233,   321,   292,     3,   636,   637,
     638,   121,   245,   365,   599,  -465,  -465,   122,   145,   292,
     293,   243,   294,   295,   296,   133,   120,   134,   297,   298,
     600,   389,   352,  -465,  -256,  -465,   390,  -465,   299,   302,
    -465,  -465,   769,   299,  -465,   371,  -465,   652,  -465,   653,
     654,   123,   655,   488,   483,   500,   181,   292,   293,   602,
     294,   295,   296,   757,  -256,   513,   372,   297,   298,   558,
    -465,   415,   304,   384,   299,   603,   390,   758,   328,   490,
     386,  -465,  -465,   560,   349,   646,  -465,  -183,   647,   355,
     648,   356,  -217,   299,  -465,  -465,  -465,  -465,  -465,  -465,
     299,  -465,  -465,  -465,  -465,   297,   298,   820,  -217,  -256,
     639,  -436,   380,   299,   821,   482,   193,  -323,   824,   361,
     376,   357,   398,   559,   399,  -217,   514,   395,  -183,   385,
     397,   822,   239,   480,   764,   604,   387,   561,   656,   676,
     785,   290,   515,   303,   418,   400,   515,   305,  -217,   682,
     290,   299,   306,   148,   785,   149,   741,   388,   750,   350,
     290,   122,  -183,   583,   205,   420,   358,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   435,
     436,   437,   438,   439,   440,   441,   442,   443,   444,   605,
     446,   447,  -323,   677,   362,   121,   150,   742,   225,   751,
     125,   190,   290,   191,   311,   290,   460,   461,   481,   290,
     686,   135,   515,   136,   181,   787,   515,   699,   247,   182,
     248,   249,   402,   466,   128,   137,   786,   140,   407,   787,
     411,   743,   125,   752,   312,   717,   468,   261,   471,   469,
     564,   262,   263,   264,   192,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   773,   234,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   696,  -436,
     313,   679,   408,   161,   409,   697,   286,   287,   162,   163,
     247,   289,   248,   249,   290,   247,   774,   248,   249,   247,
     503,   248,   249,   169,   219,   170,   220,   235,   171,   508,
     509,   153,   681,   173,   155,   511,   154,   288,   174,   156,
     682,   683,   623,   797,   624,   410,   639,   221,   307,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     288,   528,   534,   284,   285,   175,   562,   288,   286,   287,
     314,   556,   288,   286,   287,   288,   318,   286,   287,   222,
     288,   476,   626,  -273,   627,   625,   629,   288,   630,   308,
     552,   319,   616,   288,   617,   504,   567,   568,   569,   570,
     571,   572,   573,   574,   575,   576,   577,   666,   825,   322,
     642,   315,  -154,   477,   288,   247,   288,   248,   249,   353,
     354,   247,   288,   248,   249,   628,   288,   492,   493,   631,
     288,   553,   555,   288,   738,   288,   288,   288,   324,   840,
     839,   288,   288,   288,   288,   288,   288,   678,   667,   325,
     679,   288,   288,  -154,   680,   745,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   327,   329,   619,   283,   284,
     285,   748,   167,   286,   287,   739,   535,   168,   536,   286,
     287,   681,   472,   331,  -269,  -150,   333,   177,   528,   682,
     683,   670,   178,   337,   714,   347,   746,   715,   338,   537,
     538,   716,   684,   335,   592,   247,   673,   248,   249,   338,
     341,   205,   749,   672,  -269,   647,   654,   648,   655,  -153,
     342,   606,   607,   343,   344,   345,   205,   675,   346,   348,
     288,   351,   247,   359,   248,   249,   614,   473,   363,   272,
     273,   274,   360,   368,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   713,   374,   719,   381,   720,
     394,   396,   153,   286,   287,   247,   155,   248,   249,   414,
     421,   661,   448,   445,   278,   279,   280,   281,   282,   283,
     284,   285,   462,   730,   463,   464,   733,   125,   486,   736,
     286,   287,   542,   485,   543,   491,   494,   499,   505,   178,
     510,  -150,   390,   512,   581,   755,   582,   477,   288,   280,
     281,   282,   283,   284,   285,   544,   538,   586,   587,   590,
     777,   594,   595,   286,   287,   450,   451,   452,   453,   454,
     455,   456,   457,   458,   459,  -153,   596,   597,   615,   620,
     205,   726,   727,   632,   634,   645,   643,   651,   288,   659,
     288,   288,   288,   288,   288,   288,   288,   288,   288,   288,
     288,   288,   547,   288,   288,   288,   288,   288,   288,   288,
     288,   288,   288,   662,   288,   288,   664,   692,   693,   703,
     695,   704,   740,   705,   761,   706,   707,   708,   288,   288,
     709,   722,   723,   725,   288,   728,   288,   731,   734,   288,
     737,   744,   753,   754,   802,   759,   760,   810,   762,   763,
     770,   771,   290,   813,   814,   182,   815,   772,   779,   780,
     781,   782,   783,   784,   789,   808,   811,   817,   818,   819,
     837,   288,   841,   842,   843,   845,   288,   288,   846,   288,
     847,   849,   851,   465,   633,   545,   791,   800,   807,   792,
     660,   809,   793,   613,   830,   794,   498,   832,   778,   585,
     700,   844,   834,   701,   836,   801,   593,   166,   367,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     288,     0,     0,     0,     0,   288,   288,   288,   288,   288,
     288,   288,   288,   288,   288,   288,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   829,     0,     0,   831,     0,
       0,     0,     0,   833,     0,   835,     0,     0,     0,    -2,
       4,     0,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     0,    17,     0,   288,    18,     0,
       0,     0,    19,    20,     0,    21,    22,    23,    24,   848,
      25,    26,     0,    27,    28,    29,     0,    30,    31,    32,
      33,    34,    35,     0,    36,     0,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     288,     0,     0,   288,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,   288,    60,    61,    62,    63,     0,   288,   288,     0,
       0,     0,     4,     0,     5,     6,     7,     8,     9,    10,
      11,  -107,    13,    14,    15,    16,     0,    17,     0,     0,
      18,   525,   526,   527,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   228,     0,   229,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   230,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,   288,     0,   288,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,  -147,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,  -147,  -147,    21,    22,    23,    24,     0,    25,   228,
       0,   229,    28,    29,     0,     0,    31,     0,     0,     0,
       0,  -147,   230,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,  -143,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,  -143,  -143,    21,
      22,    23,    24,     0,    25,   228,     0,   229,    28,    29,
       0,     0,    31,     0,     0,     0,     0,  -143,   230,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,  -178,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,  -178,  -178,    21,    22,    23,    24,     0,
      25,   228,     0,   229,    28,    29,     0,     0,    31,     0,
       0,     0,     0,  -178,   230,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,  -174,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,  -174,
    -174,    21,    22,    23,    24,     0,    25,   228,     0,   229,
      28,    29,     0,     0,    31,     0,     0,     0,     0,  -174,
     230,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   -76,    13,    14,    15,    16,     0,    17,   519,   520,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   228,     0,   229,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   230,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,  -191,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,   547,    25,   228,
       0,   229,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   230,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,  -195,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,  -195,    25,   228,     0,   229,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   230,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,   517,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   228,     0,   229,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   230,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,   554,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,     0,    25,   228,     0,   229,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     230,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   608,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   228,     0,   229,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   230,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,   663,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   228,
       0,   229,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   230,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,   710,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,     0,    25,   228,     0,   229,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   230,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,   711,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   228,     0,   229,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   230,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,     0,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,     0,    25,   228,     0,   229,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     230,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,   721,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   -79,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   228,     0,   229,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   230,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,   826,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   228,
       0,   229,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   230,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,   827,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,     0,    25,   228,     0,   229,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   230,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,   828,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   228,     0,   229,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   230,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,  -156,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,     0,    25,   228,     0,   229,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     230,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   850,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   228,     0,   229,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   230,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,     0,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   228,
       0,   229,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   230,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     157,     0,   158,     6,     7,   130,     9,    10,    11,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   131,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   203,     0,   204,     6,     7,   130,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   131,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   210,     0,   211,     6,     7,   130,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   131,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   129,     0,     0,     6,
       7,   130,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     131,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   138,     0,
       0,     6,     7,   130,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   131,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     143,     0,     0,     6,     7,   130,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   131,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   146,     0,     0,     6,     7,   130,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   131,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   151,     0,     0,     6,     7,   130,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   131,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   164,     0,     0,     6,
       7,   130,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     131,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   185,     0,
       0,     6,     7,   130,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   131,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     434,     0,     0,     6,     7,   130,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   131,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   470,     0,     0,     6,     7,   130,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   131,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   618,     0,     0,     6,     7,   130,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   131,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   671,     0,     0,     6,
       7,   130,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     131,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   674,     0,
       0,     6,     7,   130,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   131,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     812,     0,     0,     6,     7,   130,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   131,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,   765,     0,  -104,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,     0,  -104,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   247,     0,   248,   249,
       0,     0,     0,   565,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,     0,     0,   766,   262,
     263,   264,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   566,     0,     0,     0,
       0,     0,     0,     0,   286,   287,     0,     0,   247,     0,
     248,   249,     0,     0,     0,     0,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   309,     0,
       0,   262,   263,   264,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
       0,     0,     0,     0,     0,     0,   286,   287,     0,   310,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   247,     0,   248,   249,     0,     0,     0,     0,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   316,     0,     0,   262,   263,   264,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,     0,     0,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,     0,     0,     0,     0,     0,     0,     0,     0,   286,
     287,     0,   317,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   578,   247,     0,   248,   249,     0,     0,
       0,     0,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,     0,     0,     0,   262,   263,   264,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,     0,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,     0,   246,   247,     0,   248,   249,
       0,     0,   286,   287,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,     0,     0,   579,   262,
     263,   264,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,   323,   247,     0,
     248,   249,     0,     0,   286,   287,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,     0,     0,
       0,   262,   263,   264,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,   326,
     247,     0,   248,   249,     0,     0,   286,   287,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
       0,     0,     0,   262,   263,   264,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,   330,   247,     0,   248,   249,     0,     0,   286,   287,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,     0,     0,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,   336,   247,     0,   248,   249,     0,     0,
     286,   287,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,     0,     0,     0,   262,   263,   264,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,     0,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,     0,   369,   247,     0,   248,   249,
       0,     0,   286,   287,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,     0,     0,     0,   262,
     263,   264,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,   724,   247,     0,
     248,   249,     0,     0,   286,   287,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,     0,     0,
       0,   262,   263,   264,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,   838,
     247,     0,   248,   249,     0,     0,   286,   287,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
       0,     0,     0,   262,   263,   264,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   247,     0,   248,   249,     0,     0,   286,   287,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,     0,     0,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     6,     7,   130,     9,    10,    11,     0,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    22,    23,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   195,   131,     0,    37,     0,    39,     0,     0,
      42,    43,     0,     0,    45,   196,    46,     0,    47,     0,
     197,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,    52,     6,     7,   130,
       9,    10,    11,     0,    54,    55,    56,    57,    58,    59,
       0,    60,    61,    62,    63,     0,     0,     0,     0,     0,
      22,    23,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   195,   131,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   130,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,   419,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     131,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,   187,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   130,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   131,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,   379,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,   130,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   131,     0,    37,     0,    39,     0,
       0,    42,    43,     0,   417,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,   130,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   131,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,   507,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   130,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   131,     0,
      37,     0,    39,     0,     0,    42,    43,     0,   776,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   130,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     131,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,     0,
     382,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   247,     0,   248,   249,     0,     0,   383,     0,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,     0,     0,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   382,     0,     0,     0,     0,     0,     0,     0,
     286,   287,     0,     0,   247,   563,   248,   249,     0,     0,
       0,     0,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,     0,     0,     0,   262,   263,   264,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,     0,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   610,     0,     0,     0,     0,     0,
       0,     0,   286,   287,     0,     0,   247,   611,   248,   249,
       0,     0,     0,     0,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,     0,     0,     0,   262,
     263,   264,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,   378,   247,     0,
     248,   249,     0,     0,   286,   287,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,     0,     0,
       0,   262,   263,   264,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
     247,   506,   248,   249,     0,     0,   286,   287,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
       0,     0,     0,   262,   263,   264,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   247,     0,   248,   249,     0,     0,   286,   287,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,     0,   580,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,     0,     0,   247,     0,   248,   249,
     286,   287,   612,     0,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,     0,     0,     0,   262,
     263,   264,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,     0,   247,   669,
     248,   249,     0,     0,   286,   287,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,     0,     0,
       0,   262,   263,   264,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
     247,   790,   248,   249,     0,     0,   286,   287,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
       0,     0,     0,   262,   263,   264,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   247,     0,   248,   249,     0,     0,   286,   287,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,     0,     0,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   247,     0,   248,   249,     0,     0,     0,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   262,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   247,     0,   248,   249,     0,     0,     0,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   263,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   247,     0,   248,   249,     0,     0,     0,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   264,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   247,     0,   248,   249,     0,     0,     0,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,     0,     0,     0,     0,     0,     0,
     286,   287
};

static const yytype_int16 yycheck[] =
{
       2,    14,    13,   360,    17,   245,    19,    18,    21,   347,
     584,     3,    25,    46,     4,    28,    49,   591,    31,     4,
      53,     6,     7,     8,     1,    38,    39,     1,     1,     6,
     366,   367,    45,    46,     3,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,     1,    61,    62,
       4,     5,    63,     7,     8,     9,     3,   393,     3,     1,
       3,    78,     4,     5,     6,     7,     8,     9,    44,     4,
      72,     6,     7,     1,    76,    92,     4,     0,     6,     7,
       8,    44,    56,    56,    76,    27,    28,    63,     1,     4,
       5,    93,     7,     8,     9,     1,     3,     3,    52,    53,
      92,     1,     1,    45,     3,    47,     6,    49,    98,   122,
      52,    53,    75,    98,    56,    57,    58,     1,    60,     3,
       4,     3,     6,    92,     1,   365,     1,     4,     5,    76,
       7,     8,     9,    78,    33,     1,    78,    52,    53,     1,
      82,     1,     3,     1,    98,    92,     6,    92,   150,    92,
       1,    93,    94,     1,     3,     1,    98,     3,     4,     1,
       6,     3,    62,    98,   106,   107,   108,   109,   110,   111,
      98,   113,   114,   115,   116,    52,    53,     3,    78,    78,
     537,    56,   195,    98,    10,     1,     6,     3,    55,     3,
     192,    33,     4,    55,     6,    55,    62,     3,    44,    57,
       3,   775,    46,     3,     3,     1,    57,    55,    92,     1,
       3,    78,    78,     3,   247,    27,    78,    78,    78,    45,
      78,    98,    75,     1,     3,     3,     3,    78,     3,    78,
      78,    63,    78,   473,   247,   248,    78,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,    55,
     273,   274,    78,    55,    78,    44,    44,    44,     4,    44,
       6,     1,    78,     3,     3,    78,   289,   290,    78,    78,
     584,     1,    78,     3,     1,    78,    78,   591,    56,     6,
      58,    59,   232,   306,   305,    15,    75,     1,   238,    78,
     240,    78,     6,    78,     3,   609,   319,    75,   321,   320,
      78,    79,    80,    81,    44,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,     6,     3,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,     3,    56,
       3,     6,     1,     1,     3,    10,   114,   115,     6,     7,
      56,    75,    58,    59,    78,    56,    37,    58,    59,    56,
     373,    58,    59,     1,     4,     3,     6,    44,     6,   382,
     383,     1,    37,     1,     1,   388,     6,   109,     6,     6,
      45,    46,     1,   740,     3,    44,   743,    27,     3,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     132,   403,   404,   104,   105,    33,   419,   139,   114,   115,
       3,   413,   144,   114,   115,   147,     3,   114,   115,    59,
     152,     1,     1,     3,     3,    44,     1,   159,     3,    44,
       3,    75,     1,   165,     3,   375,   449,   450,   451,   452,
     453,   454,   455,   456,   457,   458,   459,     3,   786,     3,
       1,    44,     3,    33,   186,    56,   188,    58,    59,     6,
       7,    56,   194,    58,    59,    44,   198,     6,     7,    44,
     202,    44,   412,   205,     3,   207,   208,   209,     3,   817,
     816,   213,   214,   215,   216,   217,   218,     3,    44,     3,
       6,   223,   224,    44,    10,     3,    97,    98,    99,   100,
     101,   102,   103,   104,   105,     3,     3,   520,   103,   104,
     105,     3,     1,   114,   115,    44,     1,     6,     3,   114,
     115,    37,     1,     3,     3,    10,     3,     1,   530,    45,
      46,   564,     6,     1,     3,    75,    44,     6,     6,    24,
      25,    10,    58,     3,     1,    56,   579,    58,    59,     6,
       3,   564,    44,   566,    33,     4,     4,     6,     6,    44,
       3,   501,   502,     3,     3,     3,   579,   580,     3,     3,
     302,     3,    56,     3,    58,    59,   516,    56,     3,    90,
      91,    92,    75,     3,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   608,     3,   610,    57,   612,
       3,     3,     1,   114,   115,    56,     1,    58,    59,     3,
       6,   551,     3,     6,    98,    99,   100,   101,   102,   103,
     104,   105,     3,   625,     4,     3,   628,     6,     3,   631,
     114,   115,     1,     6,     3,     3,     6,     3,     3,     6,
      57,    10,     6,    62,     3,   656,     3,    33,   380,   100,
     101,   102,   103,   104,   105,    24,    25,     3,     6,     3,
     693,     3,     3,   114,   115,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,    44,     6,     6,     3,    10,
     693,   621,   622,    10,     3,     3,    10,    10,   420,    10,
     422,   423,   424,   425,   426,   427,   428,   429,   430,   431,
     432,   433,    30,   435,   436,   437,   438,   439,   440,   441,
     442,   443,   444,     3,   446,   447,    55,     3,    56,     3,
      78,     3,    77,     6,    44,     6,     6,     6,   460,   461,
       3,     3,     3,     3,   466,     3,   468,     3,     3,   471,
       3,     3,     3,     3,   746,     3,     3,   758,    55,    55,
       3,     3,    78,   766,    55,     6,   769,     6,     3,     3,
       3,     3,     3,     3,     3,     3,     3,    75,    57,    57,
       3,   503,     3,     3,     3,     3,   508,   509,     3,   511,
       3,     3,     3,   305,   530,   406,   727,   743,   752,   729,
     549,   757,   732,   515,   796,   735,   362,   799,   695,   475,
     591,   825,   804,   591,   806,   745,   481,    31,   184,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     562,    -1,    -1,    -1,    -1,   567,   568,   569,   570,   571,
     572,   573,   574,   575,   576,   577,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   795,    -1,    -1,   798,    -1,
      -1,    -1,    -1,   803,    -1,   805,    -1,    -1,    -1,     0,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,   619,    19,    -1,
      -1,    -1,    23,    24,    -1,    26,    27,    28,    29,   839,
      31,    32,    -1,    34,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    -1,    45,    -1,    47,    48,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     672,    -1,    -1,   675,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,   713,   113,   114,   115,   116,    -1,   719,   720,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    20,    21,    22,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,   813,    -1,   815,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    44,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    44,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    44,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    17,    18,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    30,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    30,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    -1,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    -1,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,     1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,    -1,    44,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    56,    -1,    58,    59,
      -1,    -1,    -1,     1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    44,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   114,   115,    -1,    -1,    56,    -1,
      58,    59,    -1,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,     3,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   114,   115,    -1,    44,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,     3,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,
     115,    -1,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,     3,    56,    -1,    58,    59,
      -1,    -1,   114,   115,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,     3,    56,    -1,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,     3,
      56,    -1,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,     3,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,     3,    56,    -1,    58,    59,    -1,    -1,
     114,   115,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,     3,    56,    -1,    58,    59,
      -1,    -1,   114,   115,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,     3,    56,    -1,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,     3,
      56,    -1,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,     4,     5,     6,     7,     8,     9,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    27,    28,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    44,    45,    -1,    47,    -1,    49,    -1,    -1,
      52,    53,    -1,    -1,    56,    57,    58,    -1,    60,    -1,
      62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,     4,     5,     6,
       7,     8,     9,    -1,   106,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,    -1,    -1,    -1,    -1,    -1,
      27,    28,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,   102,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    57,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    57,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    55,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    57,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    55,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,    -1,
      44,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    56,    -1,    58,    59,    -1,    -1,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    56,    57,    58,    59,    -1,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    44,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    56,    57,    58,    59,
      -1,    -1,    -1,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    55,    56,    -1,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      56,    57,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    77,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,    -1,    56,    -1,    58,    59,
     114,   115,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    56,    57,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      56,    57,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     114,   115,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     114,   115
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   118,   119,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    16,    19,    23,
      24,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      38,    39,    40,    41,    42,    43,    45,    47,    48,    49,
      50,    51,    52,    53,    54,    56,    58,    60,    82,    86,
      93,    94,    98,   104,   106,   107,   108,   109,   110,   111,
     113,   114,   115,   116,   120,   121,   123,   124,   125,   128,
     129,   131,   132,   133,   136,   138,   139,   147,   148,   149,
     150,   157,   158,   159,   166,   168,   181,   183,   192,   194,
     201,   202,   203,   205,   207,   215,   216,   217,   219,   220,
     222,   225,   243,   248,   253,   257,   258,   260,   261,   263,
     264,   265,   267,   269,   273,   275,   276,   277,   278,   279,
       3,    44,    63,     3,     1,     6,   126,   127,   260,     1,
       6,    45,   263,     1,     3,     1,     3,    15,     1,   263,
       1,   260,   282,     1,   263,     1,     1,   263,     1,     3,
      44,     1,   263,     1,     6,     1,     6,     1,     3,   263,
     254,     1,     6,     7,     1,   263,   265,     1,     6,     1,
       3,     6,   218,     1,     6,    33,   221,     1,     6,   223,
     224,     1,     6,   268,   274,     1,   263,    57,   263,   280,
       1,     3,    44,     6,   263,    44,    57,    62,   263,   279,
     283,   270,   263,     1,     3,   263,   279,   263,   263,   263,
       1,     3,   279,   263,   263,   263,   263,   263,   263,     4,
       6,    27,    59,   263,   263,     4,   260,   130,    32,    34,
      45,   124,   137,   124,     3,    44,   167,   182,   193,    46,
     210,   213,   214,   124,     1,    56,     3,    56,    58,    59,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    79,    80,    81,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   114,   115,   264,    75,
      78,     1,     4,     5,     7,     8,     9,    52,    53,    98,
     122,   259,   263,     3,     3,    78,    75,     3,    44,     3,
      44,     3,     3,     3,     3,    44,     3,    44,     3,    75,
      78,    92,     3,     3,     3,     3,     3,     3,   124,     3,
       3,     3,   226,     3,   249,     3,     3,     1,     6,   255,
     256,     3,     3,     3,     3,     3,     3,    75,     3,     3,
      78,     3,     1,     6,     7,     1,     3,    33,    78,     3,
      75,     3,    78,     3,     1,    56,   271,   271,     3,     3,
       1,    57,    78,   281,     3,   134,   124,   244,    55,    57,
     263,    57,    44,    62,     1,    57,     1,    57,    78,     1,
       6,   208,   209,   272,     3,     3,     3,     3,     4,     6,
      27,   146,   146,   152,   151,   169,   184,   146,     1,     3,
      44,   146,   211,   212,     3,     1,   208,    55,   279,   102,
     263,     6,   263,   263,   263,   263,   263,   263,   263,   263,
     263,   263,   263,   263,     1,   263,   263,   263,   263,   263,
     263,   263,   263,   263,   263,     6,   263,   263,     3,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     263,   263,     3,     4,     3,   127,   263,   153,   263,   260,
       1,   263,     1,    56,   227,   228,     1,    33,   230,   250,
       3,    78,     1,     1,   259,     6,     3,     3,    92,     3,
      92,     3,     6,     7,     6,     6,     7,   122,   224,     3,
     208,   210,   210,   263,   146,     3,    57,    57,   263,   263,
      57,   263,    62,     1,    62,    78,   210,    10,   124,    17,
      18,   140,   142,   143,   145,    20,    21,    22,   124,   155,
     156,   160,   162,   164,   124,     1,     3,    24,    25,   170,
     175,   177,     1,     3,    24,   175,   185,    30,   195,   196,
     197,   198,     3,    44,    10,   146,   124,   206,     1,    55,
       1,    55,   263,    57,    78,     1,    44,   263,   263,   263,
     263,   263,   263,   263,   263,   263,   263,   263,     3,    78,
      77,     3,     3,   208,   234,   230,     3,     6,   231,   232,
       3,   251,     1,   256,     3,     3,     6,     6,     3,    76,
      92,     3,    76,    92,     1,    55,   146,   146,    10,   245,
      44,    57,    62,   209,   146,     3,     1,     3,     1,   263,
      10,   141,   144,     1,     3,    44,     1,     3,    44,     1,
       3,    44,    10,   155,     3,     1,     6,     7,     8,   122,
     179,   180,     1,    10,   176,     3,     1,     4,     6,   190,
     191,    10,     1,     3,     4,     6,    92,   199,   200,    10,
     197,   146,     3,    10,    55,   204,     3,    44,   266,    57,
     279,     1,   263,   279,     1,   263,     1,    55,     3,     6,
      10,    37,    45,    46,    58,   202,   220,   235,   236,   238,
     239,   240,     3,    56,   233,    78,     3,    10,   202,   220,
     236,   238,   252,     3,     3,     6,     6,     6,     6,     3,
      10,    10,   135,   263,     3,     6,    10,   220,   246,   263,
     263,    61,     3,     3,     3,     3,   146,   146,     3,   161,
     124,     3,   163,   124,     3,   165,   124,     3,     3,    44,
      77,     3,    44,    78,     3,     3,    44,   178,     3,    44,
       3,    44,    78,     3,     3,   260,     3,    78,    92,     3,
       3,    44,    55,    55,     3,     1,    78,   154,   229,    75,
       3,     3,     6,     6,    37,   241,    55,   279,   232,     3,
       3,     3,     3,     3,     3,     3,    75,    78,   247,     3,
      57,   140,   146,   146,   146,   173,   174,   122,   171,   172,
     180,   146,   124,   188,   189,   186,   187,   191,     3,   200,
     260,     3,     1,   263,    55,   263,   237,    75,    57,    57,
       3,    10,   202,   242,    55,   259,    10,    10,    10,   146,
     124,   146,   124,   146,   124,   146,   124,     3,     3,   210,
     259,     3,     3,     3,   247,     3,     3,     3,   146,     3,
      10,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */





/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(3) - (4)].fal_val)->type() == Falcon::Value::t_array_decl 
            )
         {
            Falcon::ArrayDecl* ad = (yyvsp[(3) - (4)].fal_val)->asArray();
            if ( (yyvsp[(1) - (4)].fal_adecl)->size() != ad->size() )
            {
               COMPILER->raiseError(Falcon::e_unpack_size );
            }
         }
         
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (1)].fal_val), new Falcon::Value() ) ) ) );
      }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );      
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
      }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 368 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 399 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 400 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 432 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 433 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, (yyvsp[(2) - (4)].fal_adecl), (yyvsp[(4) - (4)].fal_val) );
      }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );  
      }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (2)].fal_stat));
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
    }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 593 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(4) - (4)].fal_stat) != 0 )
         {
            COMPILER->addStatement( (yyvsp[(4) - (4)].fal_stat) );
         }
         
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = (yyvsp[(1) - (4)].fal_stat);
   }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (2)].fal_stat));
      
      
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 644 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 646 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 655 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 669 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 682 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 690 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 700 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 713 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 718 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 727 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 736 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 748 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 750 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 763 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 775 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 776 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 789 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 803 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 805 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 814 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 826 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 137:

/* Line 1455 of yacc.c  */
#line 837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 846 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 852 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 862 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 874 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 896 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 905 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 155:

/* Line 1455 of yacc.c  */
#line 919 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 157:

/* Line 1455 of yacc.c  */
#line 923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 944 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 956 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 978 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 998 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 1006 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 1015 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 168:

/* Line 1455 of yacc.c  */
#line 1017 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 1026 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 1032 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 1042 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 1051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 1055 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 1067 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 1077 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 1091 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 1103 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 1123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 1130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 1140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 190:

/* Line 1455 of yacc.c  */
#line 1149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 1169 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 1187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 1207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 1217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 1228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 1241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 1253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 1275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 1276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 1288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 1294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 1303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 1304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 1307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 1312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 1313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 1320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String func_name;
         if ( parent != 0 )
         {
            switch( parent->type() )
            {
            case Falcon::Statement::t_class: 
               {
                  Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
                  func_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
               }
               break;
            
            case Falcon::Statement::t_state: 
               {
                  Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );
                  func_name =  
                        stmt_state->owner()->symbol()->name() + "." + 
                        * stmt_state->name() + "#" + *(yyvsp[(2) - (2)].stringp);
               }
               break;
            
            case Falcon::Statement::t_function: 
               {
                  Falcon::StmtFunction *stmt_func = static_cast< Falcon::StmtFunction *>( parent );
                  func_name =  
                        stmt_func->name() + "##" + *(yyvsp[(2) - (2)].stringp);
               }
               break;
            
            default: 
               func_name = *(yyvsp[(2) - (2)].stringp);
                
            }
         }
         else
            func_name = *(yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 )
         {
            if( parent->type() == Falcon::Statement::t_class ) 
            {
               Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
               Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
               if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
                  COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                   cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
                   // is this a setter/getter?
                   if( ( (yyvsp[(2) - (2)].stringp)->find( "__set_" ) == 0 || (yyvsp[(2) - (2)].stringp)->find( "__get_" ) == 0 ) && (yyvsp[(2) - (2)].stringp)->length() > 6 )
                   {
                      Falcon::String *pname = COMPILER->addString( (yyvsp[(2) - (2)].stringp)->subString( 6 ));
                      Falcon::VarDef *pd = cd->getProperty( *pname );
                      if( pd == 0 )
                      {
                        pd = new Falcon::VarDef;
                        cd->addProperty( pname, pd );
                        pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                      }
                      else if( ! pd->isReflective() )
                      {
                        COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                      }
   
                   }
               }
            }
            else if ( parent->type() == Falcon::Statement::t_state ) 
            {
   
               Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );            
               if( ! stmt_state->addFunction( (yyvsp[(2) - (2)].stringp), sym ) )
               {
                  COMPILER->raiseError(Falcon::e_sm_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                  stmt_state->state()->addFunction( *(yyvsp[(2) - (2)].stringp), sym );
               }
               
               // eventually add a property where to store this thing
               Falcon::ClassDef *cd = stmt_state->owner()->symbol()->getClassDef();
               if ( ! cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) )
                  cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef );
            }
         }
         
         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 1449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), *(yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 1466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 1472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 1477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 1483 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 1492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 1497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 1507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 1510 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 1519 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 1529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 1534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 1546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 1555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 1562 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 1570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 1575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *(yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 1583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 1588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), *(yyvsp[(4) - (5)].stringp), "", false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 1593 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), *(yyvsp[(4) - (5)].stringp), "", true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 1598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 1618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 1637 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 1642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 1647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 1652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 1666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 1671 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 1676 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 1681 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 1686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 1695 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 1700 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 1707 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 1713 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 1725 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 1730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1743 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1796 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
            {
               Falcon::ClassDef *cd = cls->symbol()->getClassDef();
               if ( ! cd->inheritance().empty() )
               {
                  COMPILER->buildCtorFor( cls );
                  // COMPILER->addStatement( func ); should be done in buildCtorFor
                  // cls->ctorFunction( func ); idem
               }
            }
            COMPILER->popContext();
            //We didn't pushed a context set
            COMPILER->popFunction();
         }
      }
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 1830 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 1838 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 1839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1856 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol(*(yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 1882 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1895 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 1898 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 286:

/* Line 1455 of yacc.c  */
#line 1918 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtState* ss = static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat));
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );         
         if( ! cls->addState( static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat)) ) )
         {
            const Falcon::String* name = ss->name();
            // we're not using the state we created, afater all.
            delete ss->state();
            delete (yyvsp[(1) - (1)].fal_stat);
            COMPILER->raiseError( Falcon::e_state_adef, *name ); 
         }
         else {
            // ownership passes on to the classdef
            cls->symbol()->getClassDef()->addState( ss->name(), ss->state() );
            cls->symbol()->getClassDef()->addProperty( ss->name(), 
               new Falcon::VarDef( ss->name() ) );
         }
      }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1977 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( prop_name, def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1999 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 2031 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { 
         (yyval.fal_stat) = COMPILER->getContext(); 
         COMPILER->popContext();
      }
    break;

  case 294:

/* Line 1455 of yacc.c  */
#line 2039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
            
         COMPILER->pushContext( 
            new Falcon::StmtState( (yyvsp[(2) - (4)].stringp), cls ) ); 
      }
    break;

  case 295:

/* Line 1455 of yacc.c  */
#line 2047 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
         
         Falcon::StmtState* state = new Falcon::StmtState( COMPILER->addString( "init" ), cls );
         cls->initState( state );
         
         COMPILER->pushContext( state ); 
      }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 2068 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 300:

/* Line 1455 of yacc.c  */
#line 2079 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      }
    break;

  case 301:

/* Line 1455 of yacc.c  */
#line 2113 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 305:

/* Line 1455 of yacc.c  */
#line 2130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 307:

/* Line 1455 of yacc.c  */
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 310:

/* Line 1455 of yacc.c  */
#line 2150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( cl_name );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 311:

/* Line 1455 of yacc.c  */
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 313:

/* Line 1455 of yacc.c  */
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 317:

/* Line 1455 of yacc.c  */
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 318:

/* Line 1455 of yacc.c  */
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 321:

/* Line 1455 of yacc.c  */
#line 2262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 322:

/* Line 1455 of yacc.c  */
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 324:

/* Line 1455 of yacc.c  */
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 325:

/* Line 1455 of yacc.c  */
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 327:

/* Line 1455 of yacc.c  */
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 328:

/* Line 1455 of yacc.c  */
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( *(yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 329:

/* Line 1455 of yacc.c  */
#line 2314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 330:

/* Line 1455 of yacc.c  */
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 331:

/* Line 1455 of yacc.c  */
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 332:

/* Line 1455 of yacc.c  */
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 333:

/* Line 1455 of yacc.c  */
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 334:

/* Line 1455 of yacc.c  */
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 335:

/* Line 1455 of yacc.c  */
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 336:

/* Line 1455 of yacc.c  */
#line 2328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 337:

/* Line 1455 of yacc.c  */
#line 2329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 338:

/* Line 1455 of yacc.c  */
#line 2330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 339:

/* Line 1455 of yacc.c  */
#line 2334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 340:

/* Line 1455 of yacc.c  */
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 341:

/* Line 1455 of yacc.c  */
#line 2336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 342:

/* Line 1455 of yacc.c  */
#line 2337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 343:

/* Line 1455 of yacc.c  */
#line 2338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 344:

/* Line 1455 of yacc.c  */
#line 2339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 345:

/* Line 1455 of yacc.c  */
#line 2340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 346:

/* Line 1455 of yacc.c  */
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     }
    break;

  case 348:

/* Line 1455 of yacc.c  */
#line 2363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 349:

/* Line 1455 of yacc.c  */
#line 2364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 354:

/* Line 1455 of yacc.c  */
#line 2392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 355:

/* Line 1455 of yacc.c  */
#line 2393 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 356:

/* Line 1455 of yacc.c  */
#line 2394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 357:

/* Line 1455 of yacc.c  */
#line 2395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 358:

/* Line 1455 of yacc.c  */
#line 2396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 359:

/* Line 1455 of yacc.c  */
#line 2397 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 360:

/* Line 1455 of yacc.c  */
#line 2398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 361:

/* Line 1455 of yacc.c  */
#line 2399 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 362:

/* Line 1455 of yacc.c  */
#line 2400 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 363:

/* Line 1455 of yacc.c  */
#line 2426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 364:

/* Line 1455 of yacc.c  */
#line 2427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() * (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 365:

/* Line 1455 of yacc.c  */
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 366:

/* Line 1455 of yacc.c  */
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 367:

/* Line 1455 of yacc.c  */
#line 2488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 368:

/* Line 1455 of yacc.c  */
#line 2489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 369:

/* Line 1455 of yacc.c  */
#line 2490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 370:

/* Line 1455 of yacc.c  */
#line 2491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 371:

/* Line 1455 of yacc.c  */
#line 2492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 372:

/* Line 1455 of yacc.c  */
#line 2493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 373:

/* Line 1455 of yacc.c  */
#line 2494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 374:

/* Line 1455 of yacc.c  */
#line 2495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 375:

/* Line 1455 of yacc.c  */
#line 2496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 376:

/* Line 1455 of yacc.c  */
#line 2497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 377:

/* Line 1455 of yacc.c  */
#line 2498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 378:

/* Line 1455 of yacc.c  */
#line 2499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 379:

/* Line 1455 of yacc.c  */
#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 380:

/* Line 1455 of yacc.c  */
#line 2501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_exeq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 381:

/* Line 1455 of yacc.c  */
#line 2502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 382:

/* Line 1455 of yacc.c  */
#line 2503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 383:

/* Line 1455 of yacc.c  */
#line 2504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 384:

/* Line 1455 of yacc.c  */
#line 2505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 385:

/* Line 1455 of yacc.c  */
#line 2506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 386:

/* Line 1455 of yacc.c  */
#line 2507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 387:

/* Line 1455 of yacc.c  */
#line 2508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 388:

/* Line 1455 of yacc.c  */
#line 2509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 389:

/* Line 1455 of yacc.c  */
#line 2510 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 390:

/* Line 1455 of yacc.c  */
#line 2511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 391:

/* Line 1455 of yacc.c  */
#line 2512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 392:

/* Line 1455 of yacc.c  */
#line 2513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 393:

/* Line 1455 of yacc.c  */
#line 2514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 394:

/* Line 1455 of yacc.c  */
#line 2515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 395:

/* Line 1455 of yacc.c  */
#line 2516 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 396:

/* Line 1455 of yacc.c  */
#line 2517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 397:

/* Line 1455 of yacc.c  */
#line 2518 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 398:

/* Line 1455 of yacc.c  */
#line 2519 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 399:

/* Line 1455 of yacc.c  */
#line 2520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 406:

/* Line 1455 of yacc.c  */
#line 2528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 407:

/* Line 1455 of yacc.c  */
#line 2533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 408:

/* Line 1455 of yacc.c  */
#line 2537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 409:

/* Line 1455 of yacc.c  */
#line 2542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 410:

/* Line 1455 of yacc.c  */
#line 2548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 413:

/* Line 1455 of yacc.c  */
#line 2560 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 414:

/* Line 1455 of yacc.c  */
#line 2565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 415:

/* Line 1455 of yacc.c  */
#line 2572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 416:

/* Line 1455 of yacc.c  */
#line 2573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 417:

/* Line 1455 of yacc.c  */
#line 2574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 418:

/* Line 1455 of yacc.c  */
#line 2575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 419:

/* Line 1455 of yacc.c  */
#line 2576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 420:

/* Line 1455 of yacc.c  */
#line 2577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 421:

/* Line 1455 of yacc.c  */
#line 2578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 422:

/* Line 1455 of yacc.c  */
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 423:

/* Line 1455 of yacc.c  */
#line 2580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 424:

/* Line 1455 of yacc.c  */
#line 2581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 425:

/* Line 1455 of yacc.c  */
#line 2582 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 426:

/* Line 1455 of yacc.c  */
#line 2583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 427:

/* Line 1455 of yacc.c  */
#line 2588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 428:

/* Line 1455 of yacc.c  */
#line 2591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 429:

/* Line 1455 of yacc.c  */
#line 2594 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 430:

/* Line 1455 of yacc.c  */
#line 2597 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 431:

/* Line 1455 of yacc.c  */
#line 2600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 432:

/* Line 1455 of yacc.c  */
#line 2607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 433:

/* Line 1455 of yacc.c  */
#line 2613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 434:

/* Line 1455 of yacc.c  */
#line 2617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 435:

/* Line 1455 of yacc.c  */
#line 2618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 436:

/* Line 1455 of yacc.c  */
#line 2627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 437:

/* Line 1455 of yacc.c  */
#line 2661 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 438:

/* Line 1455 of yacc.c  */
#line 2669 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 439:

/* Line 1455 of yacc.c  */
#line 2703 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 441:

/* Line 1455 of yacc.c  */
#line 2722 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 442:

/* Line 1455 of yacc.c  */
#line 2726 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 444:

/* Line 1455 of yacc.c  */
#line 2734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 445:

/* Line 1455 of yacc.c  */
#line 2738 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 446:

/* Line 1455 of yacc.c  */
#line 2745 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 447:

/* Line 1455 of yacc.c  */
#line 2778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 448:

/* Line 1455 of yacc.c  */
#line 2794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 449:

/* Line 1455 of yacc.c  */
#line 2799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 450:

/* Line 1455 of yacc.c  */
#line 2806 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 451:

/* Line 1455 of yacc.c  */
#line 2813 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 452:

/* Line 1455 of yacc.c  */
#line 2822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 453:

/* Line 1455 of yacc.c  */
#line 2824 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 454:

/* Line 1455 of yacc.c  */
#line 2828 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 455:

/* Line 1455 of yacc.c  */
#line 2835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 456:

/* Line 1455 of yacc.c  */
#line 2837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 457:

/* Line 1455 of yacc.c  */
#line 2841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 458:

/* Line 1455 of yacc.c  */
#line 2849 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 459:

/* Line 1455 of yacc.c  */
#line 2850 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 460:

/* Line 1455 of yacc.c  */
#line 2852 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 461:

/* Line 1455 of yacc.c  */
#line 2859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 462:

/* Line 1455 of yacc.c  */
#line 2860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 463:

/* Line 1455 of yacc.c  */
#line 2864 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 464:

/* Line 1455 of yacc.c  */
#line 2865 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 467:

/* Line 1455 of yacc.c  */
#line 2872 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 468:

/* Line 1455 of yacc.c  */
#line 2878 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 469:

/* Line 1455 of yacc.c  */
#line 2885 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 470:

/* Line 1455 of yacc.c  */
#line 2886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;



/* Line 1455 of yacc.c  */
#line 7591 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 2890 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


